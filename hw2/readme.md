This program implements all base requirements with no extra credit. Below is a list of keybinds to utilize the functionality of the program. Scale for modes 1-3 was calculated through trial and error of what looked best overall. Similarly LookAt() was found through trial and error for the overall starting view. To run call 'make' followed by './hw1 heightmap/<image of choice>' in the hw1 directory.

Keybinds:
‘1’: mode 1, point rendering
‘2’: mode 2, line rendering
‘3’: mode 3, triangle rendering
‘4’: mode 4, triangle rendering with smoothing
‘t’: translate with OpenGLMatrix::Translate
‘Shift’: shift with OpenGLMatrix::Shift
Mouse movement: rotate with OpenGLMatrix::Rotate
‘a’: rotate the image for animation, toggle on and off, use in combination with ‘r’
‘r’: record 300 frames in 15 fps, displays input in modes 1-4 with varying scale and exponent values
‘9’: divide current exponent by 2, default value 1.0, only used in mode 4
‘0’: multiply current exponent by 2, default value 1.0, only used in mode 4
‘-’: divide current scale by 2, default value 1.0, only used in mode 4
‘=’: multiply current scale by 2, default value 1.0, only used in mode 4
'm': toggle z-scaling on and off, xy scaling are done with left mouse

