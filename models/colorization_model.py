import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        # Input: 1x96x96 (L channel)
        self.conv1 = self._conv_block(1, 64, 3, 1, 1)
        self.conv2 = self._conv_block(64, 128, 3, 2, 1) # 128x48x48
        self.conv3 = self._conv_block(128, 256, 3, 2, 1) # 256x24x24
        self.conv4 = self._conv_block(256, 512, 3, 2, 1) # 512x12x12
        self.conv5 = self._conv_block(512, 512, 3, 1, 1) # 512x12x12
        
    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # Input: 512x12x12
        self.deconv1 = self._deconv_block(512, 256, 4, 2, 1) # 256x24x24
        self.deconv2 = self._deconv_block(256, 128, 4, 2, 1) # 128x48x48
        self.deconv3 = self._deconv_block(128, 64, 4, 2, 1) # 64x96x96
        
        # Final layer to output 2 channels (a and b)
        self.final_conv = nn.Conv2d(64, 2, 3, 1, 1) # 2x96x96
        
    def _deconv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.final_conv(x)
        # We use Tanh to ensure the output is in the range [-1, 1]
        # which matches the normalization of our 'a' and 'b' channels
        return torch.tanh(x)

class ColorizationAutoencoder(nn.Module):
    def __init__(self):
        super(ColorizationAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        # x is the L channel (1x96x96)
        encoded = self.encoder(x)
        # decoded is the predicted ab channels (2x96x96)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == '__main__':
    # Test the model with a dummy input
    model = ColorizationAutoencoder()
    dummy_input = torch.randn(4, 1, 96, 96) # Batch size 4, 1 channel, 96x96
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Expected output shape: [4, 2, 96, 96] (Batch, a/b channels, H, W)
    assert output.shape == torch.Size([4, 2, 96, 96])
    print("Model test passed successfully.")
