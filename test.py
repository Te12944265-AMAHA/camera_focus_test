from cProfile import label
from threading import local
from black import out
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

ENTROPY = 0
VARIANCE = 1
GRADIENT = 2
AVG_EDGE_STRENGTH = 3


class CameraFocus:
    def __init__(self):
        self.path = "./camera_focus_test_images"

    def blur_and_save_image(self, imgname):
        img = cv2.imread("{}/{}".format(self.path, imgname))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_name = imgname[:-4]
        extension = imgname[-4:]
        for ksize in [15, 55, 95, 135, 355, 565]:
            blur = cv2.GaussianBlur(img, (ksize, ksize), 0)
            cv2.imwrite(
                "{}/{}_gaus_{:03}{}".format(self.path, img_name, ksize, extension), blur
            )

    def focus_entropy(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        marg = np.histogramdd(np.ravel(img), bins=10)[0] / img.size
        marg = list(filter(lambda p: p > 0, np.ravel(marg)))
        entropy = -np.sum(np.multiply(marg, np.log2(marg)))
        return entropy

    def focus_variance(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.var(img)

    def focus_image_gradient(self, image, name=""):
        "Get magnitude of gradient for given image"
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ddepth = cv2.CV_32F
        dx = cv2.Sobel(img, ddepth, 1, 0, ksize=3)
        dy = cv2.Sobel(img, ddepth, 0, 1)
        abs_grad_x = cv2.convertScaleAbs(dx)
        abs_grad_y = cv2.convertScaleAbs(dy)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        # mag = cv2.magnitude(dx, dy) / 1.414
        # mag = np.clip(mag, 0, 255)
        marg = np.histogram(np.ravel(grad), bins=np.arange(257))[0]
        # marg[:50] = np.zeros(50)
        val = grad.sum() / float(image.shape[0] * image.shape[1]) / 255.0
        plt.rcParams["figure.figsize"] = [16, 8]
        plt.subplot(1, 2, 1)
        plt.plot(
            np.arange(256),
            marg,
            label="std: {}\nmin: {}\nmax: {}\nsharpness: {}".format(
                np.std(grad), np.min(grad), np.max(grad), val
            ),
        )

        plt.legend(loc="upper left")
        plt.title(name)
        plt.subplot(1, 2, 2)
        plt.imshow(grad)
        plt.show()
        return val

    def focus_average_edge_strength(self, img):
        """
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.7.9921&rep=rep1&type=pdf
        """
        # do edge detection
        # non-max suppression
        # take average of edge
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ddepth = cv2.CV_32F
        dx = cv2.Sobel(img, ddepth, 1, 0, ksize=3)
        dy = cv2.Sobel(img, ddepth, 0, 1)
        abs_grad_x = cv2.convertScaleAbs(dx)
        abs_grad_y = cv2.convertScaleAbs(dy)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        # self.NMS(grad, k=7)
        # check range of grad, then select filter criteria
        max_grad = np.max(grad)
        min_grad = np.min(grad)
        filter_th = int((max_grad - min_grad) * 0.6) + min_grad
        grad = np.where(grad >= filter_th, grad, 0.0)
        num_valid_pix = np.sum(np.where(grad != 0, 1.0, 0.0), dtype=float)
        return np.sum(grad, dtype=float) / num_valid_pix if num_valid_pix > 0 else 0

    def NMS(self, img, k=5):
        h, w = img.shape[0], img.shape[1]
        margin = (k - 1) // 2
        for r in range(h):
            for c in range(w):
                if r < margin or r >= h - margin or c < margin or c >= w - margin:
                    img[r, c] = 0
                    continue
                local_max = 0
                for pr in range(r - margin, r + margin + 1):
                    for pc in range(c - margin, c + margin + 1):
                        if img[pr, pc] <= local_max:
                            img[pr, pc] = 0
                        else:
                            local_max = img[pr, pc]

    def test_focus(self, img_name, type=VARIANCE):
        fnames = os.listdir(self.path)
        fnames.sort()
        img_names = []
        sharpness_values = []
        for fname in fnames:
            if fname.find(img_name) != -1:
                abs_fname = "{}/{}".format(self.path, fname)
                try:
                    image = cv2.imread(abs_fname)
                except:
                    continue
                val = 0
                if type == ENTROPY:
                    val = self.focus_entropy(image)
                elif type == VARIANCE:
                    val = self.focus_variance(image)
                elif type == GRADIENT:
                    val = self.focus_image_gradient(image, name=fname)
                elif type == AVG_EDGE_STRENGTH:
                    val = self.focus_average_edge_strength(image)
                print("Image: {}\tFocus level: {}".format(fname, val))
                img_names.append(abs_fname.split("/")[-1])
                sharpness_values.append(val)
        return img_names, sharpness_values

    def create_blank(self, _width, _height, _rgb_color=(0, 0, 0)):
        "Create new image(numpy array) filled with certain color in RGB."
        image = np.zeros((_height, _width, 3), np.uint8)
        color = tuple(reversed(_rgb_color))
        image[:] = color
        return image

    def create_blank_image_file(self, imgname, width, height, rgb_color=(0, 0, 0)):
        img = self.create_blank(width, height, _rgb_color=rgb_color)
        cv2.imwrite("{}/{}".format(self.path, imgname), img)

    def generate_ground_truth_data(self, out_file):
        # read all images, compute sharpness values, write to a .txt file
        names, sharpness = self.test_focus("png", type=AVG_EDGE_STRENGTH)
        maxlen = len(max(names, key=len))
        st = ""
        for name, sp in zip(names, sharpness):
            st = st + "{}\t{:.4f}\n".format(name.ljust(maxlen, " "), sp)
        with open(out_file, "w") as f:
            f.write(st)


if __name__ == "__main__":
    camera_focus = CameraFocus()
    # camera_focus.create_blank_image_file("black.png", 2880, 1860)
    # camera_focus.test_focus("blurry", type=AVG_EDGE_STRENGTH)
    camera_focus.generate_ground_truth_data("gt.txt")
    # camera_focus.test_focus("small_ok_1")
