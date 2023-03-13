from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet.nets.pytorch_backend.conformer.encoder import Encoder as ConformerEncoder
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class ConformerSeparator(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        input_ch : int = 1,
        num_spk: int = 2,
        predict_noise: bool = False,
        adim: int = 384,
        aheads: int = 4,
        layers: int = 6,
        linear_units: int = 1536,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        normalize_before: bool = False,
        concat_after: bool = False,
        dropout_rate: float = 0.1,
        input_layer: str = "linear",
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        nonlinear: str = "relu",
        conformer_pos_enc_layer_type: str = "rel_pos",
        conformer_self_attn_layer_type: str = "rel_selfattn",
        conformer_activation_type: str = "swish",
        use_macaron_style_in_conformer: bool = True,
        use_cnn_in_conformer: bool = True,
        conformer_enc_kernel_size: int = 7,
        padding_idx: int = -1,
    ):
        """Conformer separator.

        Args:
            input_dim: input feature dimension
            num_spk: number of speakers
            predict_noise: whether to output the estimated noise signal
            adim (int): Dimension of attention.
            aheads (int): The number of heads of multi head attention.
            linear_units (int): The number of units of position-wise feed forward.
            layers (int): The number of transformer blocks.
            dropout_rate (float): Dropout rate.
            input_layer (Union[str, torch.nn.Module]): Input layer type.
            attention_dropout_rate (float): Dropout rate in attention.
            positional_dropout_rate (float): Dropout rate after adding
                                             positional encoding.
            normalize_before (bool): Whether to use layer_norm before the first block.
            concat_after (bool): Whether to concat attention layer's input and output.
                if True, additional linear will be applied.
                i.e. x -> x + linear(concat(x, att(x)))
                if False, no additional linear will be applied. i.e. x -> x + att(x)
            conformer_pos_enc_layer_type(str): Encoder positional encoding layer type.
            conformer_self_attn_layer_type (str): Encoder attention layer type.
            conformer_activation_type(str): Encoder activation function type.
            positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
            positionwise_conv_kernel_size (int): Kernel size of
                                                 positionwise conv1d layer.
            use_macaron_style_in_conformer (bool): Whether to use macaron style for
                                                   positionwise layer.
            use_cnn_in_conformer (bool): Whether to use convolution module.
            conformer_enc_kernel_size(int): Kernerl size of convolution module.
            padding_idx (int): Padding idx for input_layer=embed.
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
        """
        super().__init__()

        self._num_spk = num_spk
        self.predict_noise = predict_noise

        self.conformer = ConformerEncoder(
            idim=input_dim,
            ich = input_ch,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=linear_units,
            num_blocks=layers,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            input_layer=input_layer,
            normalize_before=normalize_before,
            concat_after=concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            macaron_style=use_macaron_style_in_conformer,
            pos_enc_layer_type=conformer_pos_enc_layer_type,
            selfattention_layer_type=conformer_self_attn_layer_type,
            activation_type=conformer_activation_type,
            use_cnn_module=use_cnn_in_conformer,
            cnn_module_kernel=conformer_enc_kernel_size,
            padding_idx=padding_idx,
        )

        num_outputs = self.num_spk + 1 if self.predict_noise else self.num_spk
        self.linear = torch.nn.ModuleList(
            #[torch.nn.Linear(adim, input_dim) for _ in range(num_outputs)]
            # complex
            [torch.nn.Linear(adim, input_dim*2) for _ in range(num_outputs)]
        )

        if nonlinear not in ("sigmoid", "relu", "tanh"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.nonlinear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
        }[nonlinear]

    def forward(
        self,
        input: torch.Tensor,
        #ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor ): Encoded feature [B, T, N]
            ilens (torch.Tensor): input lengths [Batch]
            additional (Dict or None): other data included in model
                NOTE: not used in this model

        Returns:
            masked (List[Union(torch.Tensor)]): [(B, T, N), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """

        # if complex spectrum,
        if is_complex(input):
            feature = abs(input)
        else:
            feature = input

        B,C,F,T = input.shape

        ilens = [T]*B

        # prepare pad_mask for transformer
        pad_mask = make_non_pad_mask(ilens).unsqueeze(1).to(feature.device)

        x, ilens = self.conformer(feature, pad_mask)

        masks = []
        for linear in self.linear:
            y = linear(x)
            y = self.nonlinear(y)
            masks.append(y)

        if self.predict_noise:
            *masks, mask_noise = masks
            # mask : [B, T, F*2]

        #return mask
        mask =  torch.permute(masks[0],(0,2,1))
        B, FF, T = mask.shape
        mask = torch.reshape(mask,(B,2,F,T))

        return mask

    @property
    def num_spk(self):
        return self._num_spk