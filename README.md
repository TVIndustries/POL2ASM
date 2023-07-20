# POL2ASM  

POL2ASM  

This script will take file from (what I have tested and tried) MvC2, CvS2*, and CvS1*, and covert them into:  
```
Models:  
 - .ASM files  
 - .bin files (useful for blender importing).  
Textures:  
 - .png (PNG files poor transparency retention)  
 - .tga (proper alpha retention)
 - .bin/zraw
 - .pvr (better data retnetion, use with pvrTool)
```  

## How to use

Brief description of how to use.
### Python
Have python and probably need the packages I used such as:

```Py
import os
import numpy as np
import cv2.cv2 as cv2
from PIL import Image
import struct
```  

The `import os` is for creating folders for output.  

### POL/TEX Files

Under the folder `Files` there is another folder `MvC2` (this is the game folder), put POL and TEX file inside game folder.
*: CvS1 & CvS2 need to have decompressed TEX.BIN  

### Using the script

At around `Line 154` we have the parameters:
```Py
# Parameters
game = 'MvC2_Custom'
stgID = 'STG0A'
directory = '.\\Files\\'
pol_filename = stgID + 'POL.BIN'
tex_filename = stgID + 'TEX.BIN'
game_dir = game + '\\'
texturesOnly = False
processZOrder = True
```

Here replace `game` with your game folder directory name and replace the `stgID` with your desired stage you'd wish to export.  
