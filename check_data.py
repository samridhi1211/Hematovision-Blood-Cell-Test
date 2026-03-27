from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Checking dataset...")

train_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

print("Classes:", train_data.class_indices)