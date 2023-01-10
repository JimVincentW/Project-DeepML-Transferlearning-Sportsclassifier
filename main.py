import torch
import json
import urllib
from pytorchvideo.data.encoded_video import EncodedVideo
import gradio as gr

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)


in_vid = "Skill tutorial ⚽️footballshorts footballskills football soccer.mp4"


model_name = 'x3d_s'
model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)



# Set to GPU or CPU
device = "cpu"
model = model.eval()
model = model.to(device)
json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
json_filename = "kinetics_classnames.json"
try: urllib.URLopener().retrieve(json_url, json_filename)
except: urllib.request.urlretrieve(json_url, json_filename)
with open(json_filename, "r") as f:
    kinetics_classnames = json.load(f)

# id to label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")


# model specific transforms

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
frames_per_second = 30
model_transform_params  = {
    "x3d_xs": {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 4,
        "sampling_rate": 12,
    },
    "x3d_s": {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 13,
        "sampling_rate": 6,
    },
    "x3d_m": {
        "side_size": 256,
        "crop_size": 256,
        "num_frames": 16,
        "sampling_rate": 5,
    }
}

transform_params = model_transform_params[model_name]

transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(transform_params["num_frames"]),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=transform_params["side_size"]),
            CenterCropVideo(
                crop_size=(transform_params["crop_size"], transform_params["crop_size"])
            )
        ]
    ),
)



# The duration of the input clip is also specific to the model.
clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"])/frames_per_second






def inference(in_vid):
    start_sec = 0
    end_sec = start_sec + clip_duration

    # load with and initalize EncodedVideo
    video = EncodedVideo.from_path(in_vid)

    # Load clip
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

    # transform to normalize
    video_data = transform(video_data)

    # inputs to device
    inputs = video_data["video"]
    inputs = inputs.to(device)

    # Pass through the model
    preds = model(inputs[None, ...])

    # Get predicted classes
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=5).indices[0]

    # Map the predicted classes to the label names
    pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
    return "%s" % ", ".join(pred_class_names)


inference(in_vid)
print(inference(in_vid))


#inputs = gr.inputs.Video(label="Input Video")
#outputs = gr.outputs.Textbox(label="Top 5 predicted labels")


#gr.Interface(inference, inputs, outputs, title=title, description=description, article=article, examples=examples, analytics_enabled=False).launch()