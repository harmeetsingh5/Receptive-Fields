import matplotlib.image as mpimg
import PIL.Image
import pylab as py
import numpy as np






def collect_natural_patches(num_patches = 1000000, patch_width = 8):
  """ collects image patches
  the natural images are from a specific folder of 13 .tiff files"""

  max_tries = num_patches * 50
  image_width = 200
  
  img_first_patch = 0 # the first patch number accepted from an image
  img_first_try = 0 # the first attempt to take a patch from the image
  patch_cnt = 0 # number of collected patches
  try_cnt = 0 # number of attempted collected patches
  num_pixels = patch_width * patch_width 

  patch_sample = np.empty([patch_width,patch_width],float)
  patch = np.empty([num_pixels,1],float)
  
  img_patches = np.empty([num_pixels,num_patches],float)

  # change the image sampled from
  nat_img_cnt = 1  
  image = PIL.Image.open('natural_images/' + str(nat_img_cnt) + '.tiff')
  image = np.asarray(image, 'double').transpose()  
  # normalizing the image
  image -= image.mean()
  image /= image.std()
  
  while patch_cnt < num_patches and try_cnt < max_tries:
    try_cnt += 1  # number of total patches attempted

    if (try_cnt - img_first_try) > max_tries/12 or \
      (patch_cnt - img_first_patch) > num_patches/12:
      # change the image sampled from
      nat_img_cnt += 1
      image = PIL.Image.open('natural_images/' + str(nat_img_cnt) + '.tiff')
      image = np.asarray(image, 'double').transpose()        
      # normalizing the image
      image -= image.mean()
      image /= image.std()
      
      img_first_patch = patch_cnt
      img_first_try = try_cnt
    
      # update on every switch of images
      print (int(100 * float(patch_cnt)/num_patches),' percent complete')
    
    px = np.random.randint(0,image_width - patch_width)
    py = np.random.randint(0,image_width - patch_width)
        
    patch_sample = image[px:px+patch_width,py:py+patch_width].copy()
    patch_std = patch_sample.std()
    
    if patch_std > 0.0: # > 0 to remove blank/uninteresting patches for speed
      # create the patch vector     
      patch = np.reshape(patch_sample, num_pixels)     
      patch = patch - np.mean(patch)         
      img_patches[:,patch_cnt] = patch.copy()
      patch_cnt += 1
  return img_patches

patches_mat = collect_natural_patches(num_patches = 1000000, patch_width = 8)
print('\nshape of the extracted image patch data:', patches_mat.shape)



def show_patches_mat(pre_patches, show_patch_num = 16, display=True):
  """ this function generates a 2D array to display image patches (assuming square) """
  
  patches = pre_patches
    
  tot_patches = patches.shape[1]
  data_dim = patches.shape[0]
  patch_width = int(np.round(np.sqrt(data_dim)))
  
  # extract show_patch_num patches
  disp_patch = np.zeros([data_dim, show_patch_num], float)
  for i in range(0,show_patch_num):
    patch_i = i * tot_patches // show_patch_num
  
    patch = patches[:,patch_i].copy()
    pmax  = patch.max()
    pmin = patch.min()
    # fix patch range from min to max to 0 to 1
    if pmax > pmin: 
      patch = (patch - pmin) / (pmax - pmin)
    disp_patch[:,i] = patch.copy()

  bw = 5    # border width
  pw = patch_width
  
  patches_y = int(np.sqrt(show_patch_num))
  patches_x = int(np.ceil(float(show_patch_num) / patches_y))
  patch_img = disp_patch.max() * np.ones([(pw + bw) * patches_x - bw,
    patches_y * (pw + bw) - bw], float)
  for i in range(0,show_patch_num): 
    y_i = i // patches_y
    x_i = i % patches_y
    
    reshaped = disp_patch[:,i].reshape((pw,pw))
    full_patch = np.zeros([pw, pw], float)
    full_patch[0:pw,:] = reshaped[:,:].copy()
    patch_img[x_i*(pw+bw):x_i*(pw+bw)+pw,y_i*(pw+bw):y_i*(pw+bw)+pw] = full_patch
  
  if display:
    py.bone()
    py.imshow(patch_img.T, interpolation='nearest')
    py.axis('off')
  return

icatemp = sklearn.decomposition.FastICA(n_components=50) # note, sensitive to n_components
icafit = icatemp.fit(np.transpose(patches_mat))

ica_comp = icafit.components_
# print('shape of the ica component matrix: ',ica_comp.shape)

show_patches_mat(np.transpose(ica_comp), 25)















