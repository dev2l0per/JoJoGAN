from flask import (
    Flask, request, Response, send_file, render_template
)

import torch
torch.backends.cudnn.benchmark = True
from torchvision import transforms, utils
from util import *
import os
import glob
import copy
from io import BytesIO

from distutils.util import strtobool

import numpy as np
from model import *
from e4e_projection import projection as e4e_projection

app = Flask(__name__)

latent_dim = 512

device = "cuda" if torch.cuda.is_available() else "cpu"

original_generator = Generator(1024, latent_dim, 8, 2).to(device)
ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
original_generator.load_state_dict(ckpt['g_ema'], strict=False)
mean_latent = original_generator.mean_latent(10000)

generator = copy.deepcopy(original_generator)

transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

toPILImage = transforms.ToPILImage()

pretrained_models = {
    "arcane_caitlyn": torch.load(os.path.join('models', 'arcane_caitlyn.pt'), map_location=lambda storage, loc: storage),
    "arcane_caitlyn_preserve_color": torch.load(os.path.join('models', 'arcane_caitlyn_preserve_color.pt'), map_location=lambda storage, loc: storage),
    "arcane_jinx_preserve_color": torch.load(os.path.join('models', 'arcane_jinx_preserve_color.pt'), map_location=lambda storage, loc: storage),
    "arcane_jinx": torch.load(os.path.join('models', 'arcane_jinx.pt'), map_location=lambda storage, loc: storage),
    "arcane_multi_preserve_color": torch.load(os.path.join('models', 'arcane_multi_preserve_color.pt'), map_location=lambda storage, loc: storage),
    "arcane_multi": torch.load(os.path.join('models', 'arcane_multi.pt'), map_location=lambda storage, loc: storage),
    "art": torch.load(os.path.join('models', 'art.pt'), map_location=lambda storage, loc: storage),
    "disney_preserve_color": torch.load(os.path.join('models', 'disney_preserve_color.pt'), map_location=lambda storage, loc: storage),
    "disney": torch.load(os.path.join('models', 'disney.pt'), map_location=lambda storage, loc: storage),
    "jojo_preserve_color": torch.load(os.path.join('models', 'jojo_preserve_color.pt'), map_location=lambda storage, loc: storage),
    "jojo": torch.load(os.path.join('models', 'jojo.pt'), map_location=lambda storage, loc: storage),
    "jojo_yasuho_preserve_color": torch.load(os.path.join('models', 'jojo_yasuho_preserve_color.pt'), map_location=lambda storage, loc: storage),
    "jojo_yasuho": torch.load(os.path.join('models', 'jojo_yasuho.pt'), map_location=lambda storage, loc: storage),
    "supergirl_preserve_color": torch.load(os.path.join('models', 'supergirl_preserve_color.pt'), map_location=lambda storage, loc: storage),
    "supergirl": torch.load(os.path.join('models', 'supergirl.pt'), map_location=lambda storage, loc: storage),
}


def generate(filename, pretrained):
    filepath = f'input/{filename}'
    name = strip_path_extension(filepath)+'.pt'
    aligned_face = align_face(filepath)
    my_w = e4e_projection(aligned_face, name, device).unsqueeze(0)

    ckpt = pretrained_models[pretrained]

    generator.load_state_dict(ckpt["g"], strict=False)
    seed = 3000

    torch.manual_seed(seed)
    with torch.no_grad():
        generator.eval()
        my_sample = generator(my_w, input_is_latent=True)

    return utils.make_grid(my_sample, normalize=True, range=(-1, 1))


@app.route('/jojogan', methods=['POST'])
def jojogan():
    try:
        file = request.files['file']
        pretrained = request.form['pretrained']
    except:
        return Response("Empty Field", status=400)
    
    file.save(f'input/{file.filename}')
    result = generate(file.filename, pretrained)
    result = np.squeeze(result)
    result_image = toPILImage(result)
    buffer_out = BytesIO()
    result_image.save(buffer_out, format=f'{file.content_type.split("/")[-1]}')
    buffer_out.seek(0)

    deleteFileList = glob.glob(f"input/{file.filename.split('.')[0]}.*")
    for deleteFile in deleteFileList:
        os.remove(deleteFile)

    return send_file(buffer_out, mimetype=file.content_type)

@app.route('/health', methods=['GET'])
def health_check():
    return "ok"

@app.route('/', methods=['GET'])
def main():
  return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="5000")