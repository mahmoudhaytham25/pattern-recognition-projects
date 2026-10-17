import os
import random
import shutil
from glob import glob
import torch
import torch.nn.functional as F
from torch import nn, optim, Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
from typing import Tuple, List

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Dataset roots — set env vars or place datasets under ./data/
# CONTENT_DIR: 'image-classification' Kaggle dataset (images/images)
# STYLE_DIR:   'wikiart' Kaggle dataset
RAW_CONTENT = os.environ.get('CONTENT_DIR', 'data/image-classification/images/images')
RAW_STYLE = os.environ.get('STYLE_DIR', 'data/wikiart')
WORK_BASE = os.environ.get('WORK_DIR', 'outputs')
DATA_BASE = os.path.join(WORK_BASE, 'data')

def stratified_sample_split(raw_dir, out_base, total,
                        splits=('train', 'val', 'test'),
                        ratios=(0.8, 0.1, 0.1)):
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    per_cls = total // len(classes)
    for cls in classes:
        imgs = glob(os.path.join(raw_dir, cls, '*'))
        sel = imgs if len(imgs) <= per_cls else random.sample(imgs, per_cls)
        random.shuffle(sel)
        n = len(sel)
        i1 = int(ratios[0] * n)
        i2 = i1 + int(ratios[1] * n)
        groups = {
            'train': sel[:i1],
            'val': sel[i1:i2],
            'test': sel[i2:]
        }
        for split, files in groups.items():
            d = os.path.join(out_base, split, cls)
            os.makedirs(d, exist_ok=True)
            for f in files:
                shutil.copy(f, d)

stratified_sample_split(RAW_CONTENT, os.path.join(DATA_BASE, 'content'), total=7000)
stratified_sample_split(RAW_STYLE, os.path.join(DATA_BASE, 'style'), total=3000)

print("Counts per split:")
for kind in ('content', 'style'):
    for split in ('train', 'val', 'test'):
        path = os.path.join(DATA_BASE, kind, split)
        cnt = sum(len(f) for _, _, f in os.walk(path))
        print(f"  {kind:7s} {split:5s}: {cnt}")

tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class StyleDS(Dataset):
    def __init__(self, root, split, threshold_mode='random'):
        self.C = glob(os.path.join(root, 'content', split, '*', '*'))
        self.S = glob(os.path.join(root, 'style', split, '*', '*'))
        self.threshold_mode = threshold_mode
        
    def __len__(self):
        return min(len(self.C), len(self.S))
    
    def __getitem__(self, i):
        c = Image.open(self.C[i]).convert('RGB')
        s = Image.open(self.S[i % len(self.S)]).convert('RGB')
        
        if self.threshold_mode == 'random':
            t = torch.rand(1)
        elif self.threshold_mode == 'fixed':
            t = torch.tensor([0.5])  
        else:
            t = torch.tensor([float(self.threshold_mode)])
            
        return tf(c), tf(s), t

train_dl = DataLoader(StyleDS(DATA_BASE, 'train'), batch_size=8, shuffle=True)
val_dl = DataLoader(StyleDS(DATA_BASE, 'val'), batch_size=8)
test_dl = DataLoader(StyleDS(DATA_BASE, 'test'), batch_size=4)

def adain(content_feat: Tensor, style_feat: Tensor, eps: float = 1e-5) -> Tensor:
    c_mean = content_feat.mean(dim=[2, 3], keepdim=True)
    c_std = content_feat.std(dim=[2, 3], keepdim=True) + eps
    s_mean = style_feat.mean(dim=[2, 3], keepdim=True)
    s_std = style_feat.std(dim=[2, 3], keepdim=True) + eps
    normalized = (content_feat - c_mean) / c_std
    return normalized * s_std + s_mean

class StyleNet(nn.Module):
    def __init__(self, use_vgg=True):
        super().__init__()
        
        if use_vgg:
            vgg = models.vgg19(pretrained=True).features
            self.enc = nn.Sequential(*list(vgg)[:21])
            for p in self.enc.parameters():
                p.requires_grad = False
        else:
            vit = timm.create_model('vit_base_patch16_224', pretrained=True)
            self.enc = vit
            
        self.dec = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(), nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(), nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(), nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, 3, 1, 1)
        )
        
    def forward(self, c, s, t):
        cf = self.enc(c)
        sf = self.enc(s)
        
        if isinstance(cf, dict):  # ViT returns dict
            cf = cf['x']
            sf = sf['x']
            
        tf_ = adain(cf, sf)
        feat = t.view(-1, 1, 1, 1) * tf_ + (1-t).view(-1, 1, 1, 1) * cf
        return self.dec(feat)

class VitStyleNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.encoder = vit
        
        self.adapter = nn.Sequential(
            nn.Linear(768, 512 * 4 * 4),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(), nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(), nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(), nn.Upsample(scale_factor=4),
            nn.Conv2d(64, 3, 3, 1, 1)
        )
        
    def extract_features(self, x):
        x = self.encoder.patch_embed(x)
        x = self.encoder.pos_drop(x)
        
        for blk in self.encoder.blocks:
            x = blk(x)
            
        x = self.encoder.norm(x)
        return x
    
    def forward(self, c, s, t):
        c_tokens = self.extract_features(c)
        s_tokens = self.extract_features(s)
        
        # Take CLS token
        c_cls = c_tokens[:, 0]
        s_cls = s_tokens[:, 0]
        
        # Process tokens separately for content and style
        c_feat = self.adapter(c_cls).view(-1, 512, 4, 4)
        s_feat = self.adapter(s_cls).view(-1, 512, 4, 4)
        
        # Apply AdaIN
        tf_ = adain(c_feat, s_feat)
        
        # Apply threshold
        feat = t.view(-1, 1, 1, 1) * tf_ + (1-t).view(-1, 1, 1, 1) * c_feat
        
        # Decode to image
        return self.decoder(feat)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StyleNet().to(device)
vit_model = VitStyleNet().to(device)

def gram(x: Tensor) -> Tensor:
    b, c, h, w = x.size()
    f = x.view(b, c, h*w)
    g = torch.bmm(f, f.transpose(1, 2))
    return g / (c * h * w)

def content_loss(output: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(output, target)

def style_loss(output: Tensor, target: Tensor) -> Tensor:
    g_output = gram(output)
    g_target = gram(target)
    return F.mse_loss(g_output, g_target)

def total_loss(output: Tensor, content: Tensor, style: Tensor, alpha=1.0, beta=1.0) -> Tensor:
    o = F.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)
    c = F.interpolate(content, size=(256, 256), mode='bilinear', align_corners=False)
    s = F.interpolate(style, size=(256, 256), mode='bilinear', align_corners=False)
    
    vgg = models.vgg19(pretrained=True).features[:21].to(device).eval()
    
    o_features = vgg(o)
    c_features = vgg(c)
    s_features = vgg(s)
    
    c_loss = content_loss(o_features, c_features)
    s_loss = style_loss(o_features, s_features)
    
    return alpha * c_loss + beta * s_loss

def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4, patience=5):
    optimizer = optim.Adam(model.dec.parameters(), lr=lr)
    best_val_loss = float('inf')
    counter = 0
    history = []
    
    vgg_loss = models.vgg19(pretrained=True).features[:21].to(device).eval()
    
    for epoch in range(1, num_epochs+1):
        model.train()
        train_loss = 0.0
        
        for c, s, t in train_loader:
            c, s, t = c.to(device), s.to(device), t.to(device)
            
            optimizer.zero_grad()
            output = model(c, s, t)
            
            # Compute VGG features
            c_resize = F.interpolate(c, size=(256, 256), mode='bilinear', align_corners=False)
            s_resize = F.interpolate(s, size=(256, 256), mode='bilinear', align_corners=False)
            o_resize = F.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)
            
            c_feat = vgg_loss(c_resize)
            s_feat = vgg_loss(s_resize)
            o_feat = vgg_loss(o_resize)
            
            # Content loss
            c_loss = F.mse_loss(o_feat, c_feat)
            
            # Style loss
            s_loss = F.mse_loss(gram(o_feat), gram(s_feat))
            
            # Total loss with weighting based on threshold
            loss = c_loss + s_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for c, s, t in val_loader:
                c, s, t = c.to(device), s.to(device), t.to(device)
                output = model(c, s, t)
                
                # Same loss computation as training
                c_resize = F.interpolate(c, size=(256, 256), mode='bilinear', align_corners=False)
                s_resize = F.interpolate(s, size=(256, 256), mode='bilinear', align_corners=False)
                o_resize = F.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)
                
                c_feat = vgg_loss(c_resize)
                s_feat = vgg_loss(s_resize)
                o_feat = vgg_loss(o_resize)
                
                c_loss = F.mse_loss(o_feat, c_feat)
                s_loss = F.mse_loss(gram(o_feat), gram(s_feat))
                
                loss = c_loss + s_loss
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch:02d} train {train_loss:.4f} val {val_loss:.4f}")
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_style_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return history

def evaluate_model(model, dataloader, thresholds=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
    model.eval()
    results = []
    vgg_loss_model = models.vgg19(pretrained=True).features[:21].to(device).eval()
    
    with torch.no_grad():
        for t_val in thresholds:
            content_losses = []
            style_losses = []
            
            for c, s, _ in dataloader:
                c, s = c.to(device), s.to(device)
                t_tensor = torch.full((c.size(0), 1), t_val, device=device)
                output = model(c, s, t_tensor)
                
                # Compute features for loss
                c_resize = F.interpolate(c, size=(256, 256), mode='bilinear', align_corners=False)
                s_resize = F.interpolate(s, size=(256, 256), mode='bilinear', align_corners=False)
                o_resize = F.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)
                
                c_feat = vgg_loss_model(c_resize)
                s_feat = vgg_loss_model(s_resize)
                o_feat = vgg_loss_model(o_resize)
                
                # Content loss
                c_loss = F.mse_loss(o_feat, c_feat).item()
                
                # Style loss
                s_loss = F.mse_loss(gram(o_feat), gram(s_feat)).item()
                
                content_losses.append(c_loss)
                style_losses.append(s_loss)
            
            avg_content_loss = sum(content_losses) / len(content_losses)
            avg_style_loss = sum(style_losses) / len(style_losses)
            total_loss = avg_content_loss + avg_style_loss
            
            results.append({
                'threshold': t_val,
                'content_loss': avg_content_loss,
                'style_loss': avg_style_loss,
                'total_loss': total_loss
            })
    
    return results

def generate_examples(model, dataloader, threshold=0.5, num_examples=10):
    model.eval()
    content_images = []
    style_images = []
    output_images = []
    
    with torch.no_grad():
        for c, s, _ in dataloader:
            if len(content_images) >= num_examples:
                break
                
            c, s = c.to(device), s.to(device)
            t_tensor = torch.full((c.size(0), 1), threshold, device=device)
            output = model(c, s, t_tensor)
            
            for i in range(c.size(0)):
                if len(content_images) >= num_examples:
                    break
                    
                content_images.append(c[i].cpu())
                style_images.append(s[i].cpu())
                output_images.append(output[i].cpu())
    
    return content_images, style_images, output_images

def to_image(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()

def visualize_examples(content_images, style_images, output_images):
    num_examples = len(content_images)
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, num_examples * 4))
    
    for i in range(num_examples):
        axes[i, 0].imshow(to_image(content_images[i]))
        axes[i, 0].set_title("Content Image")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(to_image(style_images[i]))
        axes[i, 1].set_title("Style Image")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(to_image(output_images[i]))
        axes[i, 2].set_title("Generated Image")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('style_transfer_examples.png')
    plt.show()

def train_all_models():
    print("Training CNN-based model...")
    cnn_history = train_model(model, train_dl, val_dl, num_epochs=20)
    
    print("\nTraining ViT-based model...")
    vit_history = train_model(vit_model, train_dl, val_dl, num_epochs=15)
    
    print("\nEvaluating CNN model across different thresholds...")
    model.load_state_dict(torch.load('best_style_model.pth'))
    cnn_results = evaluate_model(model, test_dl)
    
    print("\nEvaluating ViT model across different thresholds...")
    vit_results = evaluate_model(vit_model, test_dl)
    
    print("\nGenerating example outputs...")
    c_imgs, s_imgs, o_imgs = generate_examples(model, test_dl, threshold=0.5)
    visualize_examples(c_imgs, s_imgs, o_imgs)
    
    best_cnn_threshold = min(cnn_results, key=lambda x: x['total_loss'])['threshold']
    best_vit_threshold = min(vit_results, key=lambda x: x['total_loss'])['threshold']
    
    print(f"\nBest CNN threshold: {best_cnn_threshold} (loss: {min(cnn_results, key=lambda x: x['total_loss'])['total_loss']:.4f})")
    print(f"Best ViT threshold: {best_vit_threshold} (loss: {min(vit_results, key=lambda x: x['total_loss'])['total_loss']:.4f})")
    
    return cnn_history, vit_history, cnn_results, vit_results

from flask import Flask, request, jsonify, render_template
import io
import base64

def create_web_app(model):
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/transfer', methods=['POST'])
    def transfer_style():
        content_file = request.files['content']
        style_file = request.files['style']
        threshold = float(request.form['threshold'])
        
        content_img = Image.open(io.BytesIO(content_file.read())).convert('RGB')
        style_img = Image.open(io.BytesIO(style_file.read())).convert('RGB')
        
        content_tensor = tf(content_img).unsqueeze(0).to(device)
        style_tensor = tf(style_img).unsqueeze(0).to(device)
        threshold_tensor = torch.tensor([threshold]).view(1, 1).to(device)
        
        with torch.no_grad():
            output = model(content_tensor, style_tensor, threshold_tensor)
        
        output_img = to_image(output[0].cpu())
        output_img = (output_img * 255).astype(np.uint8)
        output_pil = Image.fromarray(output_img)
        
        buffered = io.BytesIO()
        output_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({'result': f'data:image/jpeg;base64,{img_str}'})
    
    return app

if __name__ == "__main__":
    train_history, vit_history, test_results, vit_results = train_all_models()
    
    print("\nSaving models...")
    torch.save(model.state_dict(), 'style_transfer_final.pth')
    torch.save(vit_model.state_dict(), 'vit_style_transfer_final.pth')
    
    print("\nSaving results...")
    pd.DataFrame(test_results).to_csv('cnn_threshold_results.csv', index=False)
    pd.DataFrame(vit_results).to_csv('vit_threshold_results.csv', index=False)
    
    # Uncomment to run web app
    # app = create_web_app(model)
    # app.run(debug=True, host='0.0.0.0', port=5000)
