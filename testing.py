import tensorflow as tf
from tensorflow.keras.preprocessing import image
import argparse

new_model_h5 = tf.keras.models.load_model('animal_classification_10.h5')
class_names = ['bear', 'bison', 'cat', 'cow', 'dog', 'gorilla', 'koala', 'lion', 'squirrel', 'zebra']

parser = argparse.ArgumentParser(description='Give lable of a image or result of model evaluation')
parser.add_argument('--path', help='database path')
parser.add_argument('--db', default=0, choices=[0,1], type=int)
args = parser.parse_args()

if args.db==1:
    image_path = args.path
    img = image.load_img(image_path, target_size=(160, 160, 3))
    img = image.img_to_array(img)
    img = tf.expand_dims(img, axis=0)
    predictions = new_model_h5.predict(img)
    print(class_names[tf.argmax(predictions[0])])
else:
    test_path = args.path
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        image_size=(160, 160))
    loss, accuracy = new_model_h5.evaluate(test_ds)

    print('Accuracy - ', accuracy)