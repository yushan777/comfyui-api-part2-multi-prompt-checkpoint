import json
import os
from urllib import request, parse
import random
import re

# ======================================================================
# prompt list
def get_prompt_list():

  plist = []
  plist.append("a photo of ohwx person")
  plist.append("a portrait of ohwx person in the style of Rembrandt")
  plist.append("closeup portrait of ohwx person looking up, blue bubbles in the air, cherry blossom behind, (wes anderson style:1), outside sunny day, teal and yellow colors, cinefilm, cinematic, realistic, highly detailed")
  plist.append("a charcoal sketch of ohwx person")
  return plist

# ======================================================================
# checkpoints list
def get_checkpoints_list():

  # note if in a subfolder, then you need to include subfolder
  cplist = []
  cplist.append("ohwx_person_10rpt_lr1e-5_step0500.safetensors")
  cplist.append("ohwx_person_10rpt_lr1e-5_step1000.safetensors")
  cplist.append("ohwx_person_10rpt_lr1e-5_step1500.safetensors")
  cplist.append("ohwx_person_10rpt_lr1e-5_step2000.safetensors")
  cplist.append("ohwx_person_10rpt_lr1e-5_step2500.safetensors")
  return cplist

# ======================================================================
# aspect ratio / resolutions list
def get_res_list():

  rlist = []
  rlist.append((512, 512))
  rlist.append((512, 640))
  rlist.append((512, 768))
  rlist.append((640, 512))
  rlist.append((768, 512))
  return rlist

# ======================================================================
def main():

  # get lists
  prompt_list = get_prompt_list()
  checkpoint_list = get_checkpoints_list()
  res_list = get_res_list()

  # read workflow api data from file and convert it into dictionary 
  # assign to var prompt_workflow
  prompt_workflow = json.load(open('workflow_api.json'))

  # give some easy-to-remember names to the nodes
  chkpoint_loader_node = prompt_workflow["4"]
  prompt_pos_node = prompt_workflow["6"]
  empty_latent_img_node = prompt_workflow["5"]
  ksampler_node = prompt_workflow["3"]
  save_image_node = prompt_workflow["9"]

  # set batch size in EmptyLatentImage node
  empty_latent_img_node["inputs"]["batch_size"] = 4

  # I decided to have the prompt name as the outer loop, then checkpoint, then resolution
  # for every prompt in prompt_list...
  for index, prompt in enumerate(prompt_list):

    # for every checkpoint
    for ckpt in checkpoint_list:

      # load the (next) checkpoint 
      chkpoint_loader_node["inputs"]["ckpt_name"] = ckpt  

      # set the text prompt for positive CLIPTextEncode node
      prompt_pos_node["inputs"]["text"] = prompt

      # for every resolution in the list....
      for res in res_list:
        # set image width and height
        w, h = res
        empty_latent_img_node["inputs"]["width"] = w
        empty_latent_img_node["inputs"]["height"] = h

        # set a random seed in KSampler node 
        ksampler_node["inputs"]["seed"] = random.randint(1, 18446744073709551614)

        # set prompt as folder name        
        foldername = prompt
        # (truncate to first 100 chars if necessary)
        if len(foldername) > 100:
          foldername = foldername[:100]
          # remove any special characters for safetly          
          foldername = re.sub(r'[<>:"/\\|?*]', '', foldername)
          
        ckptname = os.path.basename(ckpt) # remove path from filename
        ckptname = os.path.splitext(ckptname)[0] # remove extension
        save_image_node["inputs"]["filename_prefix"] = f'{foldername}/{ckptname}_{w}x{h}' 

        # everything set, add entire workflow to queue.
        queue_prompt(prompt_workflow)

# ======================================================================
# This function sends a prompt workflow to the specified URL and queues 
# it on the ComfyUI server running at that address.
def queue_prompt(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    req =  request.Request("http://127.0.0.1:8188/prompt", data=data)
    request.urlopen(req)    
# ======================================================================

#start main function
main()
