import cv2
import os
import numpy as np
from scipy.cluster.vq import *
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier as ovr
import pickle


def get_train_path_list(train_root_path):
    train_path_list = os.listdir(train_root_path)
    return train_path_list


def get_class_names(train_root_path, train_names):
    image_list = []
    image_class_id = []

    for index, train_class_image in enumerate(train_names):
        image_path_list = os.listdir(train_root_path + '/' + train_class_image)

        for image_path in image_path_list:
            image_list.append(train_root_path + '/' + train_class_image + '/' + image_path)
            image_class_id.append(index)

    return image_list, image_class_id


def create_descriptor_object():
    surf = cv2.xfeatures2d.SURF_create()

    return surf


def store_images_description(image_path_list, descriptor_object):
    descriptor_list = []

    for image in image_path_list:
        _, des = descriptor_object.detectAndCompute(cv2.imread(image), None)
        descriptor_list.append(des)

    # file = open('Pickle Files/descriptor_list.pickle', 'wb')
    # pickle.dump(descriptor_list, file)
    # file.close()
    # print('done')

    return descriptor_list


def stack_image_description(train_description_list):
    stacked_descriptor = train_description_list[0]

    for descriptor in train_description_list[1:]:

        stacked_descriptor = np.vstack((stacked_descriptor, descriptor))

    stacked_descriptor = np.float32(stacked_descriptor)

    # file = open('Pickle Files/stacked_descriptor.pickle', 'wb')
    # pickle.dump(stacked_descriptor, file)
    # file.close()
    # print('done')

    return stacked_descriptor


def k_means_clustering(stacked_descriptors, k=100):
    centroids, _ = kmeans(stacked_descriptors, k, 1)

    # file = open('Pickle Files/centroids.pickle', 'wb')
    # pickle.dump(centroids, file)
    # file.close()
    # print('done')

    return centroids


def calculate_histogram_of_features(image_path_list, train_description_list, centroids):
    hist_features = np.zeros((len(image_path_list), len(centroids)), 'float32')

    for i in range(0, len(image_path_list)):
        words, _ = vq(train_description_list[i], centroids)

        for w in words:
            hist_features[i][w] += 1

    # file = open('Pickle Files/hist_features.pickle', 'wb')
    # pickle.dump(hist_features, file)
    # file.close()
    # print('done')

    return hist_features


def create_standard_scaler_object(histogram_of_features):
    standard_scaler = StandardScaler().fit(histogram_of_features)

    # file = open('Pickle Files/standard_scaler.pickle', 'wb')
    # pickle.dump(standard_scaler, file)
    # file.close()
    # print('done')

    return standard_scaler


def normal_distribute(standard_scaler_object, histogram_of_features):
    normalized_hist = standard_scaler_object.transform(histogram_of_features)

    # file = open('Pickle Files/normalized_hist.pickle', 'wb')
    # pickle.dump(normalized_hist, file)
    # file.close()
    # # print('done')

    return normalized_hist


def train(normalized_histogram_of_features, image_classes_id):
    classifier = SVC(C=10, cache_size=200, class_weight='balanced', coef0=0.0,
                     decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
                     max_iter=-1, probability=False, random_state=None, shrinking=True,
                     tol=0.001, verbose=False)
    classifier.fit(normalized_histogram_of_features, np.array(image_classes_id))

    # file = open('Pickle Files/svc_classifier.pickle', 'wb')
    # pickle.dump(classifier, file)
    # file.close()
    # print('done')

    return classifier


def load_test_image(test_image_path):
    test_path_list = os.listdir(test_image_path)

    test_image = []

    for image in test_path_list:
        test_image.append(cv2.imread(test_image_path + '/' + image))

    return test_image


def store_test_image_description(test_image, descriptor_object):
    descriptor_list = []

    _, des = descriptor_object.detectAndCompute(test_image, None)
    descriptor_list.append(des)

    return descriptor_list


def calculate_test_histogram_of_features(test_description_list, centroids):
    hist_features = np.zeros((1, len(centroids)), 'float32')
    words, _ = vq(test_description_list[0], centroids)

    for w in words:
        hist_features[0][w] += 1

    return hist_features


def predict_image(classifier, test_histogram_of_features, train_names):
    id = classifier.predict(test_histogram_of_features)
    result = train_names[id[0]]

    return result


def show_result(test_image, prediction_result):
    cv2.putText(test_image, prediction_result, (0, 475), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), thickness=5)
    cv2.imshow("Result", test_image)

    if prediction_result == 'glass':
        sort = 'left'
        print('Sort Waste: ', sort)
    elif prediction_result == 'metal':
        sort = 'middle'
        print('Sort Waste: ', sort)
    elif prediction_result == 'plastic':
        sort = 'right'
        print('Sort Waste: ', sort)


if __name__ == "__main__":
    train_root_path = "wastes/train"

    train_names = get_train_path_list(train_root_path)
    image_path_list, image_classes_list = get_class_names(train_root_path, train_names)
    descriptor_object = create_descriptor_object()
    # train_description_list = store_images_description(image_path_list, descriptor_object)
    # stacked_descriptors = stack_image_description(train_description_list)
    # centroids = k_means_clustering(stacked_descriptors)
    # histogram_of_features = calculate_histogram_of_features(image_path_list, train_description_list, centroids)
    # standard_scaler_object = create_standard_scaler_object(histogram_of_features)
    # normalized_histogram_of_features = normal_distribute(standard_scaler_object, histogram_of_features)
    # classifier = train(normalized_histogram_of_features, image_classes_list)

    file = open('Pickle Files/descriptor_list.pickle', 'rb')
    train_description_list = pickle.load(file)
    file.close()

    file = open('Pickle Files/stacked_descriptor.pickle', 'rb')
    stacked_descriptors = pickle.load(file)
    file.close()

    file = open('Pickle Files/centroids.pickle', 'rb')
    centroids = pickle.load(file)
    file.close()

    file = open('Pickle Files/hist_features.pickle', 'rb')
    histogram_of_features = pickle.load(file)
    file.close()

    file = open('Pickle Files/standard_scaler.pickle', 'rb')
    standard_scaler_object = pickle.load(file)
    file.close()

    file = open('Pickle Files/normalized_hist.pickle', 'rb')
    normalized_histogram_of_features = pickle.load(file)
    file.close()

    file = open('Pickle Files/svc_classifier.pickle', 'rb')
    classifier = pickle.load(file)
    file.close()

    webcam_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = webcam_capture.read()

        if ret:
            test_description_list = store_test_image_description(frame, descriptor_object)
            test_histogram_of_features = calculate_test_histogram_of_features(test_description_list, centroids)
            test_histogram_of_features = normal_distribute(standard_scaler_object, test_histogram_of_features)
            prediction_result = predict_image(classifier, test_histogram_of_features, train_names)
            show_result(frame, prediction_result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam_capture.release()
    cv2.destroyAllWindows()