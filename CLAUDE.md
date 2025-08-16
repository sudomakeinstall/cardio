# Claude Code Configuration

## Python Coding Conventions

### Import Style
- **Prefer**: `import package as pk` over `from package import foo`
- **Use standard aliases**:
  - `import pathlib as pl`
  - `import typing as ty`
  - `import pydantic as pc`
  - `import numpy as np`
  - `import tomlkit as tk`

### Type Annotations
- **Use modern type annotations** (Python 3.9+):
  - `list[str]` instead of `List[str]`
  - `dict[str, int]` instead of `Dict[str, int]`
  - `tuple[int, str]` instead of `Tuple[int, str]`
  - `set[float]` instead of `Set[float]`

### Code Style Examples

#### Preferred Import Pattern:
```python
import pathlib as pl
import typing as ty
import pydantic as pc

def load_config(path: pl.Path) -> dict[str, ty.Any]:
    preset = pc.BaseModel.model_validate(data)
    return preset.model_dump()
```

#### Avoid:
```python
from pathlib import Path
from typing import Dict, List, Any
from pydantic import BaseModel

def load_config(path: Path) -> Dict[str, Any]:
    preset = BaseModel.model_validate(data)
    return preset.model_dump()
```

### Project-Specific Guidelines

#### VTK Imports
- Keep VTK imports explicit for clarity:
```python
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction, vtkPlanes
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkVolume
```

#### Transfer Function Management
- Use Pydantic models for validation
- Access model fields directly rather than converting to dictionaries
- Prefer `load_preset()` over wrapper functions

### File Organization
- Individual preset files in `src/cardio/assets/`
- Pydantic models for configuration validation
- Modern pathlib usage for file operations

### Error Handling
- Provide clear error messages with context
- Use `raise ... from e` for exception chaining
- Validate data at load time with Pydantic

## Volume Rendering Best Practices

### Transfer Function Blending
- Use emission-absorption model for physically accurate blending
- Order-independent blending for predictable results
- Reference academic papers in documentation

### Configuration Management
- Preset-based approach for transfer functions
- Include lighting parameters in presets
- Validate all configuration with Pydantic schemas