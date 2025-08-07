"""
Implementazione U-Net ottimizzata per rimozione stelle
Basata sull'architettura utilizzata da StarNet con miglioramenti per il nostro dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class DoubleConv(nn.Module):
    """Doppia convoluzione con normalizzazione e attivazione"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling con maxpool seguita da doppia convoluzione"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling seguita da doppia convoluzione"""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True, dropout_rate: float = 0.1):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Gestisci differenze di dimensioni
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttentionBlock(nn.Module):
    """Attention mechanism per focalizzarsi sulle stelle"""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class StarNetUNet(nn.Module):
    """
    U-Net ottimizzata per rimozione stelle
    
    Features:
    - Attention mechanism per focalizzarsi sulle stelle
    - Skip connections per preservare dettagli
    - Dropout per prevenire overfitting
    - Supporto per input multi-scala
    """
    
    def __init__(self, n_channels: int = 3, n_classes: int = 3, bilinear: bool = True, dropout_rate: float = 0.1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64, dropout_rate)
        self.down1 = Down(64, 128, dropout_rate)
        self.down2 = Down(128, 256, dropout_rate)
        self.down3 = Down(256, 512, dropout_rate)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, dropout_rate)
        
        # Attention blocks
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear, dropout_rate)
        self.up2 = Up(512, 256 // factor, bilinear, dropout_rate)
        self.up3 = Up(256, 128 // factor, bilinear, dropout_rate)
        self.up4 = Up(128, 64, bilinear, dropout_rate)
        
        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        
        # Residual connection per preservare dettagli di basso livello
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder con attention
        x4_att = self.att4(g=self.up1.up(x5), x=x4)
        x = self.up1(x5, x4_att)
        
        x3_att = self.att3(g=self.up2.up(x), x=x3)
        x = self.up2(x, x3_att)
        
        x2_att = self.att2(g=self.up3.up(x), x=x2)
        x = self.up3(x, x2_att)
        
        x1_att = self.att1(g=self.up4.up(x), x=x1)
        x = self.up4(x, x1_att)
        
        # Output con connessione residuale
        logits = self.outc(x)
        
        # Aggiungi connessione residuale pesata per preservare dettagli
        output = logits + self.residual_weight * x[:, :self.n_classes, :, :]
        
        return torch.tanh(output)  # Output in range [-1, 1]

class StarNetLoss(nn.Module):
    """
    Loss function personalizzata per rimozione stelle
    Combina L1, L2 e loss percettuale
    """
    
    def __init__(self, l1_weight: float = 1.0, l2_weight: float = 0.5, perceptual_weight: float = 0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.perceptual_weight = perceptual_weight
        
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        
        # VGG per loss percettuale (verrà inizializzato al primo uso)
        self.vgg = None
    
    def _get_vgg_features(self, x):
        """Estrae features VGG per loss percettuale"""
        if self.vgg is None:
            # Inizializza VGG solo quando necessario
            from torchvision.models import vgg16
            self.vgg = vgg16(pretrained=True).features[:16].eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
            self.vgg = self.vgg.to(x.device)
        
        return self.vgg(x)
    
    def forward(self, pred, target):
        # L1 loss (preserva dettagli)
        l1 = self.l1_loss(pred, target)
        
        # L2 loss (smoothness)
        l2 = self.l2_loss(pred, target)
        
        # Loss percettuale (preserva strutture semantiche)
        perceptual = 0
        if self.perceptual_weight > 0:
            try:
                pred_features = self._get_vgg_features(pred)
                target_features = self._get_vgg_features(target)
                perceptual = self.l2_loss(pred_features, target_features)
            except:
                perceptual = 0  # Skip se VGG non disponibile
        
        total_loss = (self.l1_weight * l1 + 
                     self.l2_weight * l2 + 
                     self.perceptual_weight * perceptual)
        
        return total_loss, {
            'l1': l1.item(),
            'l2': l2.item(), 
            'perceptual': perceptual.item() if isinstance(perceptual, torch.Tensor) else perceptual,
            'total': total_loss.item()
        }

def create_model(input_channels: int = 3, output_channels: int = 3, 
                dropout_rate: float = 0.1) -> StarNetUNet:
    """Factory function per creare il modello"""
    return StarNetUNet(
        n_channels=input_channels,
        n_classes=output_channels,
        bilinear=True,  # Usa upsampling bilineare per efficienza
        dropout_rate=dropout_rate
    )

def test_model():
    """Test del modello con input di esempio"""
    model = create_model()
    
    # Test con batch di esempio
    x = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test loss
    loss_fn = StarNetLoss()
    target = torch.randn_like(output)
    loss, loss_dict = loss_fn(output, target)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    
    # Conta parametri
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Parametri totali: {total_params:,}")
    print(f"Parametri trainabili: {trainable_params:,}")

if __name__ == "__main__":
    test_model()