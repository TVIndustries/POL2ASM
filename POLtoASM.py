import os
import numpy as np
import cv2
from PIL import Image
import struct


def save_tga_with_alpha(image, output_file):
    # Ensure the image has an alpha channel
    image_with_alpha = image.convert("RGBA")

    # Save the image as a TGA file with alpha
    image_with_alpha.save(output_file, format="TGA", transparency=0)


def create_tga_file(width, height, p_r, p_g, p_b, alpha_data, output_file):
    # TGA header
    header = struct.pack('BBBBBBBBB', 0, 0, 10, 0, 0, 0, 0, 0, 32)
    # Image width (2 bytes)
    header += struct.pack('<H', width)
    # Image height (2 bytes)
    header += struct.pack('<H', height)
    # Pixel depth (1 byte per channel)
    header += struct.pack('B', 8)
    # Image descriptor (1 byte)
    header += struct.pack('B', 0)

    # Open the output file in binary mode
    with open(output_file, 'wb') as tga_file:
        # Write the TGA header to the file
        tga_file.write(header)

        # Iterate over the RGB and alpha data
        for i in range(height):
            for j in range(width):
                # Extract the RGB and alpha values for the current pixel
                red = p_r[i][j]
                green = p_g[i][j]
                blue = p_b[i][j]
                alpha_tga = alpha_data[i][j]

                # Write the blue, green, red, and alpha values to the file
                tga_file.write(struct.pack('BBBB', blue, green, red, alpha_tga))


def signed_hex(val, nbits):
    return (val + (1 << nbits)) % (1 << nbits)


def yuv2rgb(data_param):
    y = (data_param & 0xF0)
    u = (data_param & 0x0C)
    v = (data_param & 0x02)
    blu = 1.164 * (y - 16) + 2.018 * (u - 128)
    grn = 1.164 * (y - 16) - 0.813 * (v - 128) - 0.391 * (u - 128)
    red = 1.164 * (y - 16) + 1.596 * (v - 128)
    alp = 0xFF

    return red, blu, grn, alp


def argb4444_to_rgba8888(data_param):
    # AAAA RRRR GGGG BBBB
    alp = (data_param & 0xF000) >> 12
    red = (data_param & 0x0F00) >> 8
    green = (data_param & 0x00F0) >> 4
    blue = (data_param & 0x000F)

    if alp > 0:
        alp = (alp << 4) | 0x00
    else:
        alp = 0
    blue = (blue << 4) | 0x00
    red = (red << 4) | 0x00
    green = (green << 4) | 0x00

    return red, blue, green, alp


def rgb565_to_rgba8888(data_param):
    # AAAA RRRR GGGG BBBB
    # alp = (data_param & 0xF000) >> 12
    red = (data_param & 0xF800) >> 11
    grn = (data_param & 0x07E0) >> 5
    blu = (data_param & 0x001F)

    # if alp > 0:
    #     alp = (alp << 4)
    # else:
    #     alp = 0
    alp = 0xFF
    blu = ((blu << 3) | 0x00) & 0xFF
    red = ((red << 3) | 0x00) & 0xFF
    grn = ((grn << 2) | 0x00) & 0xFF

    return red, blu, grn, alp


def bgr565_to_rgba8888(data_param):
    # AAAA RRRR GGGG BBBB
    # alp = (data_param & 0xF000) >> 12
    blue = (data_param & 0xF800) >> 11
    green = (data_param & 0x07E0) >> 5
    red = (data_param & 0x001F)

    # if alp > 0:
    #     alp = (alp << 4)
    # else:
    #     alp = 0
    alp = 0xFF
    blue = (blue << 3) | 0x00
    red = (red << 3) | 0x00
    green = (green << 2) | 0x00

    return red, blue, green, alp


def argb1555_to_rgba8888(data_param):
    # ARRR RRGG GGGB BBBB
    data_param &= 0xFFFF
    alp = (data_param & 0x8000) >> 15
    red = (data_param & 0x7C00) >> 10
    grn = (data_param & 0x03C0) >> 5
    blu = (data_param & 0x001F)

    alp = 0xFF * alp
    blu = (blu << 3)
    red = (red << 3)
    grn = (grn << 3)

    return red, blu, grn, alp


def morton1(x_param):
    x_param = x_param & 0x5555555555555555
    x_param = (x_param | (x_param >> 1)) & 0x3333333333333333
    x_param = (x_param | (x_param >> 2)) & 0x0F0F0F0F0F0F0F0F
    x_param = (x_param | (x_param >> 4)) & 0x00FF00FF00FF00FF
    x_param = (x_param | (x_param >> 8)) & 0x0000FFFF0000FFFF
    x_param = (x_param | (x_param >> 16)) & 0x00000000FFFFFFFF
    return x_param


def morton2(d):
    x_out = morton1(d)
    y_out = morton1(d >> 1)
    return x_out, y_out


# constants
sp = '    '
spD = sp + '#data '

# Parameters
game = 'MvC2'
stgID = 'STG0F'
directory = '.\\Files\\'
pol_filename = stgID + 'POL.BIN'
tex_filename = stgID + 'TEX.BIN'
game_dir = game + '\\'
texturesOnly = False
processZOrder = True

if not os.path.isfile(directory + game_dir + pol_filename):
    print("Error: File not found. [File: " + pol_filename + "]")
    exit()
else:
    print("Processing files: [%s, %s]" % (pol_filename, tex_filename))

# Directories
game_stgID = game + '_' + stgID
output_directory = '.\\Output\\' + game_dir + game_stgID + '\\'
os.makedirs(output_directory, exist_ok=True)
textureOutputDirectory = output_directory + 'Textures\\'
os.makedirs(textureOutputDirectory, exist_ok=True)
ogBinOutputDirectory = textureOutputDirectory + 'origBINs\\'
os.makedirs(ogBinOutputDirectory, exist_ok=True)
ogBinOutputDirectory = textureOutputDirectory + 'PVRs\\'
os.makedirs(ogBinOutputDirectory, exist_ok=True)
convOutputDirectory = textureOutputDirectory + 'convPNGs\\'
os.makedirs(convOutputDirectory, exist_ok=True)
modelOutputDirectory = output_directory + 'Models\\'
os.makedirs(modelOutputDirectory, exist_ok=True)
soloBinDir = modelOutputDirectory + 'soloModelBins\\'
os.makedirs(soloBinDir, exist_ok=True)
soloTexDir = modelOutputDirectory + 'soloModelBins\\Textures\\'
os.makedirs(soloTexDir, exist_ok=True)

# Getting Texture Pointer Information
pol_file = open(directory + game_dir + pol_filename, 'rb')
pol_asm_file = open(modelOutputDirectory + game_stgID + '_POL.asm', 'w+')
pol_dataSize = os.path.getsize(directory + game_dir + pol_filename)
msg_str = '; ' + game_stgID + '_POL.asm' + '\nBEG:\n'
pol_asm_file.write(msg_str)

subtractAddress = int.from_bytes(pol_file.read(4), 'little') & 0xFFFFFF00
modelCnt = int.from_bytes(pol_file.read(4), 'little')
msg_str = spD + 'ModelTable 0x%08X TextureTable TextureEnd\nModelTable:\n' % modelCnt + spD
pol_asm_file.write(msg_str)
for i in range(0, modelCnt):
    if i % 0x04 == 0 and i > 0:
        pol_asm_file.write('\n' + spD)
    msgStr = 'Model_%03d ' % i
    pol_asm_file.write(msgStr)

msgStr = '\n    #data 0x00000000 ; EndModelTable\n    #align16\n\nTextureTable:'
pol_asm_file.write(msgStr)

textureListLocation = int.from_bytes(pol_file.read(4), 'little') - subtractAddress
if textureListLocation < 0:
    print('Texture List Location does not make sense.')
    exit()
else:
    print('Texture List Location: 0x%08X' % textureListLocation)

pol_file.seek(textureListLocation + 0x08, 0)

textSubAddress = int.from_bytes(pol_file.read(4), 'little')
print('First Texture Address in DC RAM: 0x%08X' % textSubAddress)

pol_file.seek(textureListLocation, 0)
width = 0x01
height = 0x01
textureInfoList = []
texCnt = 0
while width != 0 and height != 0:
    pol_asm_file.write('\n' + spD)
    width = int.from_bytes(pol_file.read(2), 'little')
    msgStr = '0x%04X ' % width
    pol_asm_file.write(msgStr)
    if width == 0:
        continue
    height = int.from_bytes(pol_file.read(2), 'little')
    msgStr = '0x%04X ' % height
    pol_asm_file.write(msgStr)
    textureType = int.from_bytes(pol_file.read(1), 'little')
    msgStr = '0x%02X ' % textureType
    pol_asm_file.write(msgStr)
    textureFmt = int.from_bytes(pol_file.read(1), 'little')  # textureFmt Type
    msgStr = '0x%02X ' % textureFmt
    pol_asm_file.write(msgStr)
    blank_1 = int.from_bytes(pol_file.read(2), 'little')  # Blank 0x0000
    msgStr = '0x%04X ' % blank_1
    pol_asm_file.write(msgStr)
    textureLocation = int.from_bytes(pol_file.read(4), 'little')
    msgStr = '0x%08X ' % textureLocation
    pol_asm_file.write(msgStr)
    endLoc = textureLocation + (width * height * 2)
    textureLocation -= textSubAddress
    blank_2 = int.from_bytes(pol_file.read(4), 'little')  # Blank 0x00000000
    msgStr = '0x%08X ; 0x%02X, Ends @ 0x%08X' % (blank_2, texCnt, endLoc)
    pol_asm_file.write(msgStr)
    texCnt += 1
    if textureLocation < 0:
        print('Texture Location does not make sense.')
        exit()
    # print('0x%04X' % width, '0x%04X' % height, '0x%02X' % textureType, '0x%08X' % textureLocation)
    textureInformation = ('0x%04X' % width, '0x%04X' % height, '0x%02X' % textureType, '0x%02X' % textureFmt,
                          '0x%08X' % textureLocation)
    textureInfoList.append(textureInformation)
msgStr = '0x0000 0x00 0x00 0x0000 0x00000000 0x00000000 ; END\n\nTextureEnd:\n'
pol_asm_file.write(msgStr)

for i in range(0, modelCnt):
    msgStr = 'Model_%03d:\n    #import_raw_data \"soloModelBins/%s_Model_%03d.bin\"\n' % (i, game_stgID, i)
    pol_asm_file.write(msgStr)
msgStr = 'STG_END:\n    #data 0x00000000\n    #align16'
pol_asm_file.write(msgStr)
# Getting Model Pointer Information
if not texturesOnly:
    pol_file.seek(0, 0)
    modelPointerListLocation = int.from_bytes(pol_file.read(4), 'little') - subtractAddress
    numberOfModels = int.from_bytes(pol_file.read(4), 'little')
    pol_file.seek(modelPointerListLocation, 0)
    modelPointerList = []
    for i in range(numberOfModels):
        curPointer = int.from_bytes(pol_file.read(4), 'little') - subtractAddress
        nextPointerPos = pol_file.tell()

        nextPointer = int.from_bytes(pol_file.read(4), 'little')
        if nextPointer > subtractAddress:
            nextPointer -= subtractAddress
        else:
            # print('Next Pointer: 0x%08X' % signed_hex(nextPointer, 32))
            nextPointer = pol_dataSize
            # print('    Setting Next Pointer: 0x%08X' % signed_hex(pol_dataSize, 32))

        pointerDiff = nextPointer - curPointer
        pol_file.seek(nextPointerPos, 0)
        item = (curPointer, nextPointer, pointerDiff)
        modelPointerList.append(item)
        # print('Model %02d Information:\n  Current Pointer: 0x%08X , Next Pointer: 0x%08X, Difference: 0x%08X\n'
        print('Model %02d Information:\n  curPtr: 0x%08X, nxtPtr: 0x%08X, ptDiff: 0x%08X\n'
              % (i, curPointer, nextPointer, pointerDiff))

    print('\n+ Beginning to output isolated model data...')

    curFileHeader = '{1:s}_Model_{0:03d}.bin'
    for count, pointerInformation in enumerate(modelPointerList):
        # print('Model %02d @ 0x%08X' % (count, pointerInformation[0]))
        curModelName = curFileHeader.format(count, game_stgID)

        begPos = pointerInformation[0]
        endPos = pointerInformation[1]
        ptDist = pointerInformation[2]
        pol_file.seek(begPos, 0)

        pulledData = []
        countedDWORDs = 0

        curFileName = soloBinDir + curModelName
        print(' -> Exporting:', curModelName)
        with open(curFileName, 'wb') as curFile:
            for i in range(0, ptDist, 4):
                value = int.from_bytes(pol_file.read(4), 'little')
                curFile.write(value.to_bytes(4, 'little', signed=False))
        # print('Bytes Calculated for Model %02d is 0x%08X bytes.' % (count, ptDist))

print('\n+ Beginning to output isolated texture data...')
fileList = []
tex_file = open(directory + game_dir + tex_filename, 'rb')
tFNS = tex_filename.split('.')
zero = 0
curFileHeader = 'TexID_{0:03d}-' + tFNS[0] + '-TexType_{1:s}-TexFmt_{2:s}.pvr'
shortFileHeader = 'TexID_{0:03d}'
# print('(width, height, textureType, textureLocation)')
#          [   P,    V,    R,    T,     E,     Z,    I,    S,     CF,     TF,   0x00, 0x00,     TH,  WID,    GHT,  HEI ]
# type_0 = [0x50, 0x56, 0x52, 0x54,   0x08, 0x20, 0x00, 0x00,   0x01,   0x01,   0x00, 0x00,   0x40, 0x00,   0x40, 0x00 ]
# type_1 = [0x50, 0x56, 0x52, 0x54,   0x08, 0x48, 0x00, 0x00,   0x01,   0x03,   0x00, 0x00,   0x00, 0x01,   0x00, 0x01 ]
# type_2 = [0x50, 0x56, 0x52, 0x54,   0x08, 0x18, 0x00, 0x00,   0x01,   0x03,   0x00, 0x00,   0x80, 0x00,   0x80, 0x00 ]
# type_3 = [0x50, 0x56, 0x52, 0x54,   0x08, 0x20, 0x00, 0x00,   0x01,   0x01,   0x00, 0x00,   0x40, 0x00,   0x40, 0x00 ]

for count, curTextureInfo in enumerate(textureInfoList):
    curTextureName = curFileHeader.format(count, curTextureInfo[2], curTextureInfo[3])
    print('  -> Exporting:', curTextureName)
    print(curTextureInfo)

    fileList.append(curTextureName)
    width = int(curTextureInfo[0], 16)
    height = int(curTextureInfo[1], 16)
    colorFmt = int(curTextureInfo[2], 16)
    textureFmt = int(curTextureInfo[3], 16)
    texLoc = int(curTextureInfo[4], 16)
    tex_file.seek(texLoc, 0)
    dataSize = (width * height)
    remaining_pvr_datasize = dataSize * 2 + 8
    pvrt_array = [0x50, 0x56, 0x52, 0x54]

    with open(ogBinOutputDirectory + curTextureName, 'wb') as curFile:
        for p in pvrt_array:
            curFile.write(p.to_bytes(1, 'little', signed=False))
        curFile.write(remaining_pvr_datasize.to_bytes(4, 'little', signed=False))
        curFile.write(colorFmt.to_bytes(1, 'little', signed=False))
        curFile.write(textureFmt.to_bytes(1, 'little', signed=False))
        curFile.write(zero.to_bytes(2, 'little', signed=False))
        curFile.write(width.to_bytes(2, 'little', signed=False))
        curFile.write(height.to_bytes(2, 'little', signed=False))
        for i in range(dataSize):
            data = int.from_bytes(tex_file.read(2), 'little')
            curFile.write(data.to_bytes(2, 'little', signed=False))

# print(fileList)
if processZOrder:
    print('\n+ Beginning to output un-zordered and converted texture data...')
    zrawTextureOutputDirectory = textureOutputDirectory + 'unzdRAWs\\'
    os.makedirs(zrawTextureOutputDirectory, exist_ok=True)
    for count, file in enumerate(fileList):
        print('  -> Processing:', file)
        colorFormat = textureInfoList[count][2]
        # print('    info ::', ogBinOutputDirectory + file)
        dataSize = os.path.getsize(ogBinOutputDirectory + file) - 0x10
        pixelAmount = round(dataSize / 2)
        print('    info ::', '{0:d}[0x{0:08X}]'.format(pixelAmount),
              'pixels, sqrt -> {0:d}[0x{0:08X}]'.format(round(np.sqrt(pixelAmount))))
        width = round(np.sqrt(pixelAmount))
        imgArr = [[0 for x in range(width)] for y in range(width)]
        inFile = open(ogBinOutputDirectory + file, 'rb')
        inFile.seek(0x10, 0)
        for i in range(0, pixelAmount):
            x, y = morton2(i)
            imgArr[x][y] = int.from_bytes(inFile.read(2), 'little')
            # print(morton2(i), ' = ', hex(imgArr[x][y]))
        inFile.close()
        fns = file.split('.')
        out_z_filename = fns[0] + '.raw'
        outFile = open(zrawTextureOutputDirectory + out_z_filename, 'wb+')
        print('    info ::', out_z_filename)
        for j in range(0, width):
            for k in range(0, width):
                # print('(',j,',',k,')', hex(imgArr[k][j]))
                outFile.write(imgArr[k][j].to_bytes(2, 'little', signed=False))
        outFile.close()
        outFile = open(zrawTextureOutputDirectory + out_z_filename, 'rb')
        swap_paired_rows = False
        dataSize = os.path.getsize(zrawTextureOutputDirectory + out_z_filename)
        pixelAmount = round(dataSize / 2)
        # print('    info ::', '{0:d}[0x{0:08X}]'.format(pixelAmount),
        #       'pixels, sqrt -> {0:d}[0x{0:08X}]'.format(round(np.sqrt(pixelAmount))))
        w = round(np.sqrt(pixelAmount))
        h = w
        half_h = h >> 1
        numWords = w * h
        r = np.zeros((w, h), 'uint8')
        g = np.zeros((w, h), 'uint8')
        b = np.zeros((w, h), 'uint8')
        alpha = np.zeros((w, h), 'uint8')
        rgbaArray = np.zeros((w, h, 4), 'uint8')
        rgbaArray[..., 0] = r
        rgbaArray[..., 1] = g
        rgbaArray[..., 2] = b
        rgbaArray[..., 3] = alpha

        fid = open(zrawTextureOutputDirectory + out_z_filename, 'rb')

        for wx in range(0, w):
            for hy in range(0, h):
                if colorFormat == '0x02':
                    data = int.from_bytes(fid.read(2), 'little')
                    r[wx][hy], b[wx][hy], g[wx][hy], alpha[wx][hy] = argb4444_to_rgba8888(data)
                elif colorFormat == '0x01':
                    data = int.from_bytes(fid.read(2), 'little')
                    r[wx][hy], b[wx][hy], g[wx][hy], alpha[wx][hy] = rgb565_to_rgba8888(data)
                elif colorFormat == '0x00':
                    data = int.from_bytes(fid.read(2), 'little')
                    r[wx][hy], b[wx][hy], g[wx][hy], alpha[wx][hy] = argb1555_to_rgba8888(data)
                elif colorFormat == 'BGR565':
                    data = int.from_bytes(fid.read(2), 'little')
                    r[wx][hy], b[wx][hy], g[wx][hy], alpha[wx][hy] = bgr565_to_rgba8888(data)
                elif colorFormat == 'YUV422':
                    data = int.from_bytes(fid.read(1), 'little')
                    r[wx][hy], b[wx][hy], g[wx][hy], alpha[wx][hy] = yuv2rgb(data)
        fid.close()
        # print(np.shape(r[1][:]))
        if swap_paired_rows:
            for i in range(0, h, 2):
                temp = np.copy(r[i][:])
                temp2 = np.copy(r[i + 1][:])
                r[i][:] = temp2
                r[i + 1][:] = temp
                temp = np.copy(b[i][:])
                temp2 = np.copy(b[i + 1][:])
                b[i][:] = temp2
                b[i + 1][:] = temp
                temp = np.copy(g[i][:])
                temp2 = np.copy(g[i + 1][:])
                g[i][:] = temp2
                g[i + 1][:] = temp

        img_RGBA = cv2.merge((b, g, r, alpha))
        # img_tga_b = cv2.cvtColor(img_RGBA, cv2.COLOR_BGR2RGB)
        # Flip the image vertically.
        flipped_image_vertical = cv2.flip(img_RGBA, 0)

        # Flip the image horizontally
        flipped_image_horizontal = cv2.flip(img_RGBA, 1)

        # Rotate the image by 90 degrees clockwise
        # ROTATE_90_COUNTERCLOCKWISE, ROTATE_90_CLOCKWISE
        rotated_image = cv2.rotate(img_RGBA, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Display the original and transformed images

        flipped_rotated_image_vertical = cv2.flip(rotated_image, 0)
        # cv2.imshow('Original Image', img_RGBA)
        # cv2.imshow('Flipped Image (Vertical)', flipped_image_vertical)
        # cv2.imshow('Flipped Image (Horizontal)', flipped_image_horizontal)
        # cv2.imshow('Rotated Image (90 degrees CW)', rotated_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        out_z_fns = out_z_filename.split('.')
        out_conv_filename = out_z_fns[0]
        shortName = out_conv_filename.split('-')
        print('    info ::', out_conv_filename + '.png')
        print('    info ::', shortName[0] + '.png')
        # print('    info ::', convOutputDirectory + out_conv_filename + '_PNG.png') #
        # create_tga_file(w, h, r, g, b, alpha, soloTexDir + shortName[0] + '.tga')

        image_tga = Image.fromarray(cv2.cvtColor(flipped_rotated_image_vertical, cv2.COLOR_BGRA2RGBA))
        # image_tga.save(soloTexDir + shortName[0] + '.tga', format='TGA', transparency=0)
        # cv2.imwrite(soloTexDir + shortName[0] + '.png', flipped_rotated_image_vertical)
        cv2.imwrite(convOutputDirectory + out_conv_filename + '.png', flipped_rotated_image_vertical)

        # Save the TGA file with alpha
        save_tga_with_alpha(Image.fromarray(cv2.cvtColor(flipped_rotated_image_vertical, cv2.COLOR_BGRA2RGBA)), soloTexDir + shortName[0] + '.tga')

        print('')

dirPath = soloTexDir

for file in os.listdir(dirPath):
    try:
        img = Image.open(f"{dirPath}/{file}")
        r, g, b, a = img.split()
        a.save(f"{dirPath}/{file.split('.')[0]}-alpha.png")
    except ValueError:
        print(f"Image {file} doesn't have Alpha channel")
