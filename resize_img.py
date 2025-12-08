import cv2
import os

def resize_img_to_indicated_ratio(src_dir_name, file_name, extend_name, dst_dir_name):
    os.makedirs(dst_dir_name, exist_ok=True)
    file_path = f"{src_dir_name}/{file_name}.{extend_name}"
    img = cv2.imread(f"{file_path}")
    h, w, c = img.shape

    min_ratio = 0.1
    max_ratio = 1.0
    div = 100
    one_div = (max_ratio - min_ratio) / div

    ### 不能縮太小, 至少要 model input size 320 所以從 5 開始
    for go_div in range(div):
        ratio =  min_ratio + one_div * go_div
        resize_w = round(w * ratio)
        resize_h = round(h * ratio)
        min_len = min(resize_w, resize_h)
        dst_path = f"{dst_dir_name}/{file_name}x%.03f.jpg" % ratio 
        if(min_len >= 320):
            img_resize = cv2.resize(img, (resize_w, resize_w))
            # cv2.imshow("img_resize", img_resize)
            # cv2.waitKey(0)
            cv2.imwrite(dst_path, img_resize)
            print(dst_path, "finish")
        else:
            print(dst_path, "too small")

src_dir_name = "Doc3D/"
dst_dir_name = "Doc3D/samples"
extend_name = "jpg"
file_name = "01--1_1_1-pr_Page_141-PZU0001"
file_name = "phone_clear_template14"
resize_img_to_indicated_ratio(src_dir_name, file_name, extend_name, dst_dir_name)


# src_dir_name = "."
# dst_dir_name = "original"
# extend_name = "jpg"

# file_name = "20250918_1m_00000"
# resize_img_to_indicated_ratio(src_dir_name, file_name, extend_name, dst_dir_name)
# file_name = "20250920_1.2m_00001"
# resize_img_to_indicated_ratio(src_dir_name, file_name, extend_name, dst_dir_name)
# file_name = "20250920_1.4m_00002"
# resize_img_to_indicated_ratio(src_dir_name, file_name, extend_name, dst_dir_name)
# file_name = "20251105_00009"
# resize_img_to_indicated_ratio(src_dir_name, file_name, extend_name, dst_dir_name)
# file_name = "20251105_00010"
# resize_img_to_indicated_ratio(src_dir_name, file_name, extend_name, dst_dir_name)

# extend_name = "png"
# file_name = "20251014_00000"
# resize_img_to_indicated_ratio(src_dir_name, file_name, extend_name, dst_dir_name)
# file_name = "20251014_00001"
# resize_img_to_indicated_ratio(src_dir_name, file_name, extend_name, dst_dir_name)