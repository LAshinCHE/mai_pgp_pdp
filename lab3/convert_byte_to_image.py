from PIL import Image
import struct
import ctypes

input_filename = input("input filename: ")
output_filename = input("output filename: ")

fin = open(input_filename, 'rb')
(w, h) = struct.unpack('ii', fin.read(8))
buff = ctypes.create_string_buffer(4 * w * h)
fin.readinto(buff)
fin.close()
img = Image.new('RGBA', (w, h))
pix = img.load()
offset = 0
for j in range(h):
	for i in range(w):
		(r, g, b, a) = struct.unpack_from('cccc', buff, offset)
		pix[i, j] = (ord(r), ord(g), ord(b), ord(a))
		offset += 4
img.save(output_filename)