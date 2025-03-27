import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh


def get_face_landmarks(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            return [(int(l.x * image.shape[1]), int(l.y * image.shape[0])) for l in landmarks]
    return None


def apply_delaunay_warping(source_img, target_img, source_points, target_points):
    rect = cv2.boundingRect(np.array(target_points))
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(target_points)

    triangles = subdiv.getTriangleList().astype(np.int32)
    warped_image = np.zeros_like(target_img)

    for t in triangles:
        pt1, pt2, pt3 = (t[:2], t[2:4], t[4:6])

        index1 = target_points.index(min(target_points, key=lambda p: np.linalg.norm(np.array(p) - np.array(pt1))))
        index2 = target_points.index(min(target_points, key=lambda p: np.linalg.norm(np.array(p) - np.array(pt2))))
        index3 = target_points.index(min(target_points, key=lambda p: np.linalg.norm(np.array(p) - np.array(pt3))))

        src_tri = np.float32([source_points[index1], source_points[index2], source_points[index3]])
        tgt_tri = np.float32([target_points[index1], target_points[index2], target_points[index3]])

        matrix = cv2.getAffineTransform(src_tri, tgt_tri)
        warped_triangle = cv2.warpAffine(source_img, matrix, (target_img.shape[1], target_img.shape[0]))

        mask = np.zeros_like(target_img, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32([tgt_tri]), (1, 1, 1))
        warped_image = warped_image * (1 - mask) + warped_triangle * mask

    return warped_image


def overlay_faces(source_img, target_img):
    if source_img is None or target_img is None:
        print("Error: One or both images are not uploaded.")
        return None

    source_img = cv2.resize(source_img, (target_img.shape[1], target_img.shape[0]))

    source_landmarks = get_face_landmarks(source_img)
    target_landmarks = get_face_landmarks(target_img)

    if source_landmarks is None or target_landmarks is None:
        print("Ошибка: не удалось обнаружить лица.")
        return None

    warped_face = apply_delaunay_warping(source_img, target_img, source_landmarks, target_landmarks)

    blended = cv2.addWeighted(target_img, 0.5, warped_face, 0.5, 0)
    return blended

source_path = "face7.jpg"
target_path = "face8.jpg"

source_img = cv2.imread(source_path)
target_img = cv2.imread(target_path)

result = overlay_faces(source_img, target_img)

if result is not None:
    cv2.imwrite("merged_face.jpg", result)
    cv2.imshow("Merged Face", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()