from vtkmodules.vtkRenderingCore import vtkRenderer


class Object:
    def __init__(self, cfg: str, renderer: vtkRenderer):
        self.label: str = cfg["label"]
        self.directory: str = cfg["directory"]
        self.suffix: str = cfg["suffix"]
        self.visible: bool = cfg["visible"]
        self.renderer = renderer

    def path_for_frame(self, frame: int) -> str:
        return f"{self.directory}/{frame}.{self.suffix}"
