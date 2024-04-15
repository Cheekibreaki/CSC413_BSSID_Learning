import tensorflow as tf

floor_model = tf.keras.models.load_model('Floor_model/floor_model(AE_256_128)-Conv(99-22,66-22,33-22).h5')
location_model = tf.keras.models.load_model('Location_model/Location_model.h5')

print("Floor Model Summary:")
floor_model.summary()

print("\nLocation Model Summary:")
location_model.summary()

sae_model = tf.keras.models.load_model('AE_model/AE_256_128.h5')
building_model = tf.keras.models.load_model('Building_model/building_model(AE_256_128)-3.h5')

print("SAE Model Summary:")
sae_model.summary()

print("\nBuilding Model Summary:")
building_model.summary()