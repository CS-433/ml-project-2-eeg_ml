import os
import sys
import torch
import numpy as np

from lib.guided_backprop import GuidedBackPropagation, run_backprop
from lib.classifiers import classifier_CNN

from lib.mapper import load_xyz, map_function
from lib.options import Options


'''
X_dat = XY[:,0]
Y_dat = XY[:,1]
# create x-y points to be used in heatmap
num_points = 2000
xi = np.linspace(X_dat.min(), X_dat.max(), num_points)
yi = np.linspace(Y_dat.min(), Y_dat.max(), num_points)

dataset = torch.load('data/training_data/EEG_dataset.pth')
test_idx = torch.load('data/training_data/splits.pth')['splits'][0]['test']
means = dataset["means"][0].type(torch.FloatTensor)
stddevs = dataset["stddevs"][0].type(torch.FloatTensor)

EEGs = dataset['dataset'][test_idx].type(torch.FloatTensor)
labels = dataset['genders'][test_idx].type(torch.LongTensor)

net = classifier_CNN(in_channel=128, num_points=300, n_class=2)

save_path_female = 'mid_results/grad/female'
save_path_male = 'mid_results/grad/male'
if not os.path.exists(save_path_female):
    os.makedirs(save_path_female)
if not os.path.exists(save_path_male):
    os.makedirs(save_path_male)

# Run pre-trained CNN on test trail and obtain gradient by Guided Backprop.
# Gradient are saved at ./mid_results/grad.
for fold in range(5):
    model_load_path = 'checkpoints/random/CNN_split_%d_best.pth'%(fold)
    net.load_state_dict(torch.load(model_load_path))
    net.eval()
    bp = GuidedBackPropagation(model=net)

    for i in range(len(EEGs)):
        eeg = (EEGs[i] - means)/stddevs
        target_class = labels[i]

        eeg = eeg.t().unsqueeze(0)
        pred = bp.forward(eeg)
        correct = (pred==target_class).item()
        bp.backward(ids=target_class)
        guided_grads = bp.generate() # (1, 300, 128)
        guided_grads = guided_grads[0].t().numpy() # (128, 300)
        save_path = save_path_female if target_class else save_path_male
        array_name = 'fold_%d_trail_%04d_%s.npy'%(fold, i, str(correct))
        np.save(os.path.join(save_path, array_name), guided_grads)

# Load saved gradient and average over trails.
for gender in ['female', 'male']:
    grad_path = 'mid_results/grad/' + gender
    grad_files = os.listdir(grad_path)

    list_grad_sign_clip = []
    for f in grad_files:
        print(f)
        correct = f.split('.')[0].split('_')[-1]
        if correct == 'False':
            continue
            
        trail = int(f.split('.')[0].split('_')[3])
        eeg = (EEGs[trail] - means)/stddevs
        eeg = eeg.numpy()

        grad = np.load(os.path.join(grad_path, f))
        grad_sign = grad*np.sign(eeg)
        grad_sign_clip = np.clip(grad_sign, 0, None)

        list_grad_sign_clip.append(grad_sign_clip)

    grad_sign_clip = sum(list_grad_sign_clip)/len(list_grad_sign_clip)
    np.save('mid_results/grad/' + gender + '_grad_sign_clip.npy', grad_sign_clip)


# Plot significance heat map.
plt.rcParams["figure.figsize"] = 30,15
x = np.linspace(-50,250, num=300)/512*1000

grad_sign_clip_f = np.load('mid_results/grad/female_grad_sign_clip.npy')
grad_sign_clip_m = np.load('mid_results/grad/male_grad_sign_clip.npy')


grad_sign_clip = np.abs(grad_sign_clip_f + grad_sign_clip_m)

fig, ax = plt.subplots()
extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
ax.imshow(grad_sign_clip, cmap="magma", aspect="auto", extent=extent)
ax.set_yticks([])
ax.set_xlim(-90, 480)

plt.ylabel('Channel', fontsize=45)
plt.xlabel('Time (ms)', fontsize=45)
plt.xticks(fontsize=45)
plt.yticks(fontsize=45)
plt.tight_layout()
plt.savefig('figs/heat-sign.png')

if opt.gif:
    save_path = 'mid_results/grad/gif'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    EEGs_f_sample = EEGs[labels==1].mean(dim=0)
    EEGs_m_sample = EEGs[labels==0].mean(dim=0)

    f_max = EEGs_f_sample.max()
    m_max = EEGs_m_sample.max()
    f_min = EEGs_f_sample.min()
    m_min = EEGs_m_sample.min()

    grad_sign_clip = np.abs(grad_sign_clip_f + grad_sign_clip_m)
    grad_max = grad_sign_clip.max()
    grad_min = grad_sign_clip.min()

    images = []
    for t in range(len(grad_sign_clip[0])):
        print(t)
        heat = cv2.imread(os.path.join(save_path, 'heat_'+str(t)+'.png'))
        eeg_m = cv2.imread(os.path.join(save_path, 'eeg_m_'+str(t)+'.png'))
        eeg_f = cv2.imread(os.path.join(save_path, 'eeg_f_'+str(t)+'.png'))

        img = cv2.hconcat([eeg_f, heat, eeg_m])[:,:,::-1]
        img = imageio.core.util.Array(img)
        images.append(img)

    imageio.mimsave('figs/topographic.gif', images, fps=60)
'''

def main(model_load_path, dataset_path, splits_path, fig_path, save_gif, save_path='./mid_results/grad-tmp'):

    dataset = torch.load(dataset_path)
    test_idx = torch.load(splits_path)['splits'][0]['test']
    means = dataset["means"][0].type(torch.FloatTensor)
    stddevs = dataset["stddevs"][0].type(torch.FloatTensor)

    EEGs = dataset['dataset'][test_idx].type(torch.FloatTensor)
    labels = dataset['genders'][test_idx].type(torch.LongTensor)

    net = classifier_CNN(in_channel=128, num_points=300, n_class=2)

    if save_gif:
        try:
            xyz = np.load('data/xyz.npy')
            XY = map_function(xyz)
            print('Load xyz array succeed!')
        except:
            print('Load xyz array fail! Recompute and save!')
            xyz_file = 'data/Biosemi128OK.xyz'
            xyz, channel_name = load_xyz(xyz_file)
            np.save('data/xyz.npy', xyz)
    else:
        XY = None

    save_path = './tmp/grad'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    run_backprop(net, model_load_path, EEGs, labels, means, stddevs, save_path, fig_path, plot_gif=save_gif, XY=XY)

if __name__ == "__main__":
    opt, _ = Options().parse(if_print=False)

    save_gif = opt.gif
    dataset_path = opt.eeg_dataset
    splits_path = opt.splits_path
    fig_path = opt.fig_path
    model_load_path = opt.load_path
    main(model_load_path, dataset_path, splits_path, fig_path, save_gif)