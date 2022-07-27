import os.path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import StereoDataset_new
import time
from model import DSC_stereo_compression


# Train parameters:
# paths to KITTI dataset folder
stereo_dir_2012 = '/home/access/disk/dev/data_sets/kitti/Sharons datasets/data_stereo_flow_multiview'
stereo_dir_2015 = '/home/access/disk/dev/data_sets/kitti/Sharons datasets/data_scene_flow_multiview'


batch_size = 2
lr_start = 1e-4
log_interval = 50
epoch_patience = 16
n_epochs = 100
n_epochs_training_base_ae = 10
M = 32   # cut images to multiple of 32 (spatial down-sampling is *32)

start_from_pretrained = ''   # Option: path to start from pretrained point - (model,optimizer,scheduler,epoch)
save_path = ''               # Option: specify save path for the model

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# Data transforms
tsfm = transforms.Compose([transforms.ToTensor()])
tsfm_val = transforms.Compose([transforms.CenterCrop((320, 1224)), transforms.ToTensor()])

training_data = StereoDataset_new(stereo_dir_2012, stereo_dir_2015, isTrainingData=True, randomFlip=True,
                                  RandomCrop=True, colorJitter=True, transform=tsfm)
val_data = StereoDataset_new(stereo_dir_2012, stereo_dir_2015, isTrainingData=False, transform=tsfm_val)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=20)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


# Load model:
model = DSC_stereo_compression(n_ch_comp=8)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=epoch_patience, verbose=True)

train_losses = []
train_counter = []
val_losses = []
val_counter = [i * len(train_dataloader.dataset) for i in range(n_epochs + 1)]

epoch_start = 1
if start_from_pretrained != '':
    checkpoint = torch.load(start_from_pretrained)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch_start = checkpoint['epoch']

model.train()


lossL1 = torch.nn.L1Loss()
# Epochs
best_loss = 10000
best_val_loss = 10000

def eval(best_val_loss, epoch):
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch, data in enumerate(val_dataloader):
            # Get stereo pair
            images_cam1, images_cam2 = data
            # Cut to be multiple of 32 (M)
            shape = images_cam1.size()
            images_cam1 = images_cam1[:, :, :M * (shape[2] // M), :M * (shape[3] // M)]
            images_cam2 = images_cam2[:, :, :M * (shape[2] // M), :M * (shape[3] // M)]
            images_cam1 = images_cam1.to(device)
            images_cam2 = images_cam2.to(device)

            # get model outputs
            _, mse_2, _, _ = model(images_cam1, images_cam2)
            loss = mse_2  # only final rec loss
            val_loss += loss.item()  # * images_cam1.size(0)
    model.train()
    val_loss = val_loss / len(val_dataloader)
    val_losses.append(val_loss)
    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(save_path, 'model_bestVal_loss.pth'))

    return best_val_loss, val_loss


best_val_loss,_ = eval(best_val_loss, 0)
for epoch in range(epoch_start, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0

    # Training
    epoch_start_time = time.time()
    for batch_idx, data in enumerate(train_dataloader):
        # Get stereo pair
        images_cam1, images_cam2 = data
        # Cut to be multiple of 32 (M)
        shape = images_cam1.size()
        images_cam1 = images_cam1[:, :, :M * (shape[2] // M), :M * (shape[3] // M)]
        images_cam2 = images_cam2[:, :, :M * (shape[2] // M), :M * (shape[3] // M)]
        images_cam1 = images_cam1.to(device)
        images_cam2 = images_cam2.to(device)

        optimizer.zero_grad()

        loss_base_rec, loss_final_rec, _, _ = model(images_cam1, images_cam2)

        if epoch < n_epochs_training_base_ae:
            loss = loss_base_rec + loss_final_rec
        else:
            loss = loss_final_rec
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if batch_idx % log_interval == 0:
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * batch_size) + ((epoch - 1) * len(train_dataloader.dataset)))

    train_loss = train_loss / len(train_dataloader)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, save_path+'model_weights.pth')

    best_val_loss, val_loss = eval(best_val_loss, epoch)
    print('Epoch: {} \tTraining Loss: {:.6f}\tVal Loss: {:.6f}\tEpoch Time: {:.6f}'
          .format(epoch, train_loss, val_loss, time.time() - epoch_start_time))


# Plot train & test loss curves
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(val_counter, val_losses, color='red')
plt.legend(['Train Loss', 'Val Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('L1 loss')
plt.show()

print("Done!")




