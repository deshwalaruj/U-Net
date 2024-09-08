import torch
import torch.nn as nn
from loguru import logger


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_layers(x)
        
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features = [64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.in_channels=in_channels, 
        self.out_channels=out_channels, 
        self.features=features
        
        self.encoder_conv_layers = nn.ModuleList()

        self.decoder_conv_layers = nn.ModuleList()
        self.decoder_upsample_layers = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        for idx, num_channels in enumerate(features):
            self.encoder_conv_layers.append(DoubleConv(in_channels=in_channels, out_channels=num_channels))
            in_channels = num_channels # for next iter setting input_channels to current out channels    
            
        
        self.bridge = DoubleConv(in_channels=features[-1], out_channels=features[-1])
        
        for idx, num_channels in enumerate(reversed(features)):
            
            self.decoder_upsample_layers.append(
                nn.ConvTranspose2d(
                    in_channels=2*num_channels if idx!=0 else num_channels,
                    out_channels=num_channels,
                    kernel_size=2,
                    stride=2
                )
            )
            self.decoder_conv_layers.append(
                DoubleConv(
                    in_channels=num_channels*2, 
                    out_channels=num_channels,
                ) 
            )
        
        self.final_layer = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        skip_connections = []
        for idx in range(len(self.encoder_conv_layers)):
            x = self.encoder_conv_layers[idx](x)
            skip_connections.append(x)
            logger.info(f"Skip connection layers size : {x.shape}")
            x = self.pool(x)
        
        x = self.bridge(x)

        for idx in range(len(self.decoder_conv_layers)):
            logger.info(f"Shape of X : {x.shape}")
            up_sample = self.decoder_upsample_layers[idx](x)
            logger.info(f"Shape after upsample : {up_sample.shape}")
            logger.info(f"Correspndong skip connection shape :  {skip_connections[-(idx + 1)].shape}")
            concat_x = torch.cat((up_sample, skip_connections[-(idx + 1)]), dim=1)
            logger.info(f"Size after concat : {concat_x.shape}")
            x = self.decoder_conv_layers[idx](concat_x)
        

        return self.final_layer(x)
            

def test():

    x = torch.randn(1, 3, 64, 64)

    model = UNet(in_channels=3, out_channels=1, features=[4, 8, 16, 32])
    output = model(x)

    print(f"Input shape : {x.shape}, Output shape : {output.shape}")                                           
            
if __name__ == "__main__":
    test()
