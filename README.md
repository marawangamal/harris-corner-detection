
## Project Description

Harris Corner Detection implementation. Instantiate Harris_Corner_Detector Class and call disp_corners method for use. Cornerness equation used can be found here: https://courses.cs.washington.edu/courses/cse576/06sp/notes/HarrisDetector.pdf


## Sample Output

<table style="width:100%">
  <tr>
  <img src="https://github.com/mgamal96/Keypoint-Detection/blob/master/building_kps.jpg?raw=true" width="600">
  </tr>
</table>

## Usage

```python


# 1. Import image
img = cv2.imread("building.jpg")

# 2. Instatiate detector object
hcd = Harris_Corner_Detector(sigma =3, corner_thresh=0.01, k=0.04)

# 3a. Obtain corners overlaid onto image
img_corners = hcd.disp_corners(img)
cv2.imwrite("building_kps.jpg", img_corners)

# 3b. Obtain corners
corners = hcd.get_corners(img)

```
