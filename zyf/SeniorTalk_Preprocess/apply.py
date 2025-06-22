from wavaugmentate.mcs import MultiChannelSignal as Mcs
from wavaugmentate.aug import SignalAugmentation as Aug


# File name of original sound.
file_name = "./wav/train/S0001/Elderly0122S0001W0003.wav"

# Create Mcs-object.
mcs = Mcs()

# Read WAV-file to Mcs-object.
mcs.read(file_name)

# Change quantity of channels to 7.
mcs.split(7)

# Create augmentation object.
aug = Aug(mcs)

# Apply delays.
# Corresponds to channels quantity.
delay_list = [0, 150, 200, 250, 300, 350, 400]
aug.delay_ctrl(delay_list)

# Apply amplitude changes.
# Corresponds to channels quantity.
amplitude_list = [1, 0.17, 0.2, 0.23, 0.3, 0.37, 0.4]
aug.amplitude_ctrl(amplitude_list)
sound_aug_file_path="./test.wav"
# Augmentation result saving by single file, containing 7 channels.
aug.get().write(sound_aug_file_path)

# Augmentation result saving to 7 files, each 1 by channel.
# ./outputwav/sound_augmented_1.wav
# ./outputwav/sound_augmented_2.wav and so on.
aug.get().write_by_channel(sound_aug_file_path)
