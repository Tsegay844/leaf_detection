import json
import torch
import os
import onnx
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from onnxsim import simplify
from ultralytics import YOLO
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import (
    de_parallel,
    select_device,
    smart_inference_mode,
)

from ppq.api import load_native_graph, espdl_quantize_onnx
from ppq.executor import TorchExecutor
from ppq import QuantizationSettingFactory
from nn.modules.esp_head import ESPDetect
from nn.modules import *

class CaliDataset(Dataset):
    def __init__(self, path, img_shape=224):
        super().__init__()
        height, width = img_shape if isinstance(img_shape, (list, tuple)) else (img_shape, img_shape)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((height, width)),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ]
        )

        self.imgs_path = []
        self.path = path
        for img_name in os.listdir(self.path):
            img_path = os.path.join(self.path, img_name)
            self.imgs_path.append(img_path)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img = Image.open(self.imgs_path[idx])
        img = self.transform(img)
        return img


def quant(imgsz):
    BATCH_SIZE = 32
    INPUT_SHAPE = [3, *imgsz] if isinstance(imgsz, (list, tuple)) else [3, imgsz, imgsz]
    DEVICE = "cpu"
    TARGET = "esp32p4"
    NUM_OF_BITS = 8
    ONNX_PATH = "path/to/your/onnx"
    ESPDL_MODLE_PATH = "espdet_pico_224_224_mycat.espdl"
    CALIB_DIR = "path/to/your/calibration/dataset"
    
    # load model
    model = onnx.load(ONNX_PATH)
    sim = True
    if sim:
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx.shape_inference.infer_shapes(model), ONNX_PATH)

    calibration_dataset = CaliDataset(CALIB_DIR, img_shape=imgsz)
    dataloader = DataLoader(
        dataset=calibration_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    def collate_fn(batch: torch.Tensor) -> torch.Tensor:
        return batch.to(DEVICE)

    # default setting
    quant_setting = QuantizationSettingFactory.espdl_setting()
    # set to your quantization setting

    # For example, Equalization
    quant_setting.equalization = True
    quant_setting.equalization_setting.iterations = 10
    quant_setting.equalization_setting.value_threshold = 0.3
    quant_setting.equalization_setting.opt_level = 2
    quant_setting.equalization_setting.interested_layers = None
 

    quant_ppq_graph = espdl_quantize_onnx(
        onnx_import_file=ONNX_PATH,
        espdl_export_file=ESPDL_MODLE_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,
        input_shape=[1] + INPUT_SHAPE,
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        collate_fn=collate_fn,
        setting=quant_setting,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=False,
        verbose=0,
        inputs=None,
    )
    return quant_ppq_graph


class QuantizedModelValidator(BaseValidator):
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None, executor=None):
        """Executes validation process, running inference on dataloader and computing performance metrics."""
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (
                trainer.epoch == trainer.epochs - 1
            )
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml"):
                LOGGER.warning(
                    "WARNING ⚠️ validating an untrained model YAML will result in 0 mAP."
                )
            callbacks.add_integration_callbacks(self)

            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )

            # self.model = model
            self.device = model.device  # update device

            self.args.half = model.fp16  # update half

            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine

            imgsz = check_imgsz(self.args.imgsz, stride=stride)

            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = model.metadata.get(
                    "batch", 1
                )  # export.py models default to batch-size 1
                LOGGER.info(
                    f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})"
                )

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(
                    emojis(
                        f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"
                    )
                )

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = (
                    0  # faster CPU val as time dominated by inference, not dataloading
                )

            if not pt:
                self.args.rect = False  # set to false

            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(
                self.data.get(self.args.split), self.args.batch
            )

            model.eval()
            model.warmup(
                imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz)
            )  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val

        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)
            # Inference
            #################################################################################
            #################################################################################
            with dt[1]:
                preds = ppq_graph_inference(executor, "detect", batch["img"], "cpu")
            #################################################################################
            #################################################################################
            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]
            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(
            zip(
                self.speed.keys(),
                (x.t / len(self.dataloader.dataset) * 1e3 for x in dt),
            )
        )
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")

        LOGGER.info(
            "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                *tuple(self.speed.values())
            )
        )
        if self.args.save_json and self.jdict:
            with open(str(self.save_dir / "predictions.json"), "w") as f:
                LOGGER.info(f"Saving {f.name}...")
                json.dump(self.jdict, f)  # flatten and save
            stats = self.eval_json(stats)  # update stats
        if self.args.plots or self.args.save_json:
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
        return stats


def make_quant_validator_class(executor):
    class QuantDetectionValidator(DetectionValidator):
        def __init__(
            self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None
        ):
            super().__init__(dataloader, save_dir, pbar, args, _callbacks)
            self.executor = executor

        def __call__(self, trainer=None, model=None):
            return QuantizedModelValidator.__call__(self, trainer, model, self.executor)
    return QuantDetectionValidator


def ppq_graph_init(quant_func, imgsz, device, native_path=None):
    """
    Init ppq graph inference.
        # case 1: PTQ graph validation: ppq_graph = quant_func()
        # case 2: QAT graph validation:
                    utilize .native to load the graph
                    while training, the .native model is saved along with .espdl model
    """
    if native_path is not None:
        ppq_graph = load_native_graph(native_path)
    else:
        ppq_graph = quant_func(imgsz)

    executor = TorchExecutor(graph=ppq_graph, device=device)
    return executor

# remember to change the input of ppq_graph_inference function in QuantizedModelValidator
def ppq_graph_inference(executor, task, inputs, device):
    """ppq graph inference"""
    graph_outputs = executor(inputs)
    if task == "detect":
        x = [
            torch.cat((graph_outputs[i], graph_outputs[i + 1]), 1)
            for i in range(0, 6, 2)
        ]
        detect_model = ESPDetect(nc=1, ch=[32, 64, 128]) #set to your own nc
        detect_model.stride = [8.0, 16.0, 32.0]
        detect_model.to(device)
        y = detect_model._inference(x)
        return y
    else:
        raise NotImplementedError(f"{task} is not supported.")


if __name__ == "__main__":
    # load model.pt to enter val method and thereby run BaseGraph inference
    model = YOLO("path/to/best.pt")
    # eval quantized model
    executor = ppq_graph_init(quant, imgsz=224,  device="cpu", native_path=None)
    QuantDetectionValidator = make_quant_validator_class(executor)
    results = model.val(
        data="cfg/datasets/your_yaml_name.yaml",
        split="val",
        imgsz=224,
        device="cpu",
        validator=QuantDetectionValidator,
        save_json=False,
        save=True,
    )    
