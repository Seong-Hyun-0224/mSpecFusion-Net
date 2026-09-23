# ITERATIVE MODEL
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.decomposition import PCA

class SimpleAugmentation(layers.Layer):
    def __init__(self):
        super(SimpleAugmentation, self).__init__()

    def call(self, x, training=False):
        if training:
            # Random horizontal flip
            x = tf.image.random_flip_left_right(x)
            # Random vertical flip
            x = tf.image.random_flip_up_down(x)
            
            # Random rotation by 0, 90, 180, 270 degrees
            k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)  # 0~3 중 랜덤
            x = tf.image.rot90(x, k=k)
        return x

class DatasetSplitter:
    """A utility to split input dataset dynamically based on given keys."""
    def __init__(self, sp_channels=8, psp_channels=8, uv_channels=1, rgb_channels=3, prgb_channels=3):
        self.sp_channels = sp_channels
        self.psp_channels = psp_channels
        self.uv_channels = uv_channels
        self.rgb_channels = rgb_channels
        self.prgb_channels = prgb_channels
        self.total_channels = sp_channels + psp_channels + uv_channels + rgb_channels + prgb_channels

    def splitData(self, data, name_type_dataset):
        """
        Splits the data into subsets based on the provided keys.

        Args:
            data: Input tensor of shape (batch_size, height, width, total_channels, 1).
            keys: A list of keys indicating the required data splits.
                  E.g., ['sp', 'psp+uv', 'sp+uv']

        Returns:
            A dictionary where keys are the same as input keys, and values are the corresponding split data.
        """
        split_data = {}
        if name_type_dataset == 'rgb':
            data[..., :self.rgb_channels, :]
            split_data[name_type_dataset] = tf.squeeze(data[..., :self.rgb_channels, :], axis=-1)
        elif name_type_dataset == 'sp':
            start = self.rgb_channels
            split_data[name_type_dataset] = data[..., start:start + self.sp_channels, :]
            # chs = [3,4,5,6,7,8,9,10]
        elif name_type_dataset == 'sp_3':
            start = self.rgb_channels
            sp_origin = data[..., start:start + self.sp_channels, :]
            data_2 = sp_origin[..., 2:3, :]
            data_3 = sp_origin[..., 3:4, :]
            data_4 = sp_origin[..., 4:5, :]
            split_data[name_type_dataset] = tf.concat([data_2, data_3, data_4], axis=3)
        
        elif name_type_dataset == 'sp_4':
            start = self.rgb_channels
            sp_origin = data[..., start:start + self.sp_channels, :]
            data_2 = sp_origin[..., 2:3, :]
            data_3 = sp_origin[..., 3:4, :]
            data_4 = sp_origin[..., 4:5, :]
            data_5 = sp_origin[..., 5:6, :]
            split_data[name_type_dataset] = tf.concat([data_2, data_3, data_4, data_5], axis=3)

        elif name_type_dataset == 'sp_5':
            start = self.rgb_channels
            sp_origin = data[..., start:start + self.sp_channels, :]
            data_2 = sp_origin[..., 2:3, :]
            data_3 = sp_origin[..., 3:4, :]
            data_4 = sp_origin[..., 4:5, :]
            data_5 = sp_origin[..., 5:6, :]
            data_6 = sp_origin[..., 6:7, :]
            split_data[name_type_dataset] = tf.concat([data_2, data_3, data_4, data_5, data_6], axis=3)

        elif name_type_dataset == 'sp_6':
            start = self.rgb_channels
            sp_origin = data[..., start:start + self.sp_channels, :]
            data_0 = sp_origin[..., 0:1, :]
            data_2 = sp_origin[..., 2:3, :]
            data_3 = sp_origin[..., 3:4, :]
            data_4 = sp_origin[..., 4:5, :]
            data_5 = sp_origin[..., 5:6, :]
            data_6 = sp_origin[..., 6:7, :]
            split_data[name_type_dataset] = tf.concat([data_0, data_2, data_3, data_4, data_5, data_6], axis=3)

        elif name_type_dataset == 'sp_7':
            start = self.rgb_channels
            sp_origin = data[..., start:start + self.sp_channels, :]
            data_0 = sp_origin[..., 0:1, :]
            data_1 = sp_origin[..., 1:2, :]
            data_2 = sp_origin[..., 2:3, :]
            data_3 = sp_origin[..., 3:4, :]
            data_4 = sp_origin[..., 4:5, :]
            data_5 = sp_origin[..., 5:6, :]
            data_6 = sp_origin[..., 6:7, :]
            split_data[name_type_dataset] = tf.concat([data_0, data_1, data_2, data_3, data_4, data_5, data_6], axis=3)
        elif name_type_dataset == 'uv':
            start = self.rgb_channels + self.sp_channels
            split_data[name_type_dataset] = data[..., start:start + self.uv_channels, 0]  # Remove last dimension for 2D Conv
        elif name_type_dataset == 'prgb':
            start = self.rgb_channels + self.sp_channels + self.uv_channels
            split_data[name_type_dataset] =  tf.squeeze(data[..., start:start + self.prgb_channels, :], axis=-1)
        elif name_type_dataset == 'psp':
            start = self.rgb_channels + self.sp_channels + self.uv_channels + self.prgb_channels
            split_data[name_type_dataset] = data[..., start:start + self.psp_channels, :]
            # chs = [15,16,17,18,19,20,21,22]
        elif name_type_dataset == 'psp_3':
            start = self.rgb_channels + self.sp_channels + self.uv_channels + self.prgb_channels
            psp_origin = data[..., start:start + self.sp_channels, :]
            data_3 = psp_origin[..., 3:4, :]
            data_4 = psp_origin[..., 4:5, :]
            data_7 = psp_origin[..., 7:8, :]
            split_data[name_type_dataset] = tf.concat([data_3, data_4, data_7], axis=3)

        elif name_type_dataset == 'psp_4':
            start = self.rgb_channels + self.sp_channels + self.uv_channels + self.prgb_channels
            psp_origin = data[..., start:start + self.sp_channels, :]
            data_0 = psp_origin[..., 0:1, :]
            data_3 = psp_origin[..., 3:4, :]
            data_4 = psp_origin[..., 4:5, :]
            data_7 = psp_origin[..., 7:8, :]
            split_data[name_type_dataset] = tf.concat([data_0, data_3, data_4, data_7], axis=3)

        elif name_type_dataset == 'psp_5':
            start = self.rgb_channels + self.sp_channels + self.uv_channels + self.prgb_channels
            psp_origin = data[..., start:start + self.sp_channels, :]
            data_0 = psp_origin[..., 0:1, :]
            data_2 = psp_origin[..., 2:3, :]
            data_3 = psp_origin[..., 3:4, :]
            data_4 = psp_origin[..., 4:5, :]
            data_7 = psp_origin[..., 7:8, :]
            split_data[name_type_dataset] = tf.concat([data_0, data_2, data_3, data_4, data_7], axis=3)

        elif name_type_dataset == 'psp_6':
            start = self.rgb_channels + self.sp_channels + self.uv_channels + self.prgb_channels
            psp_origin = data[..., start:start + self.sp_channels, :]
            data_0 = psp_origin[..., 0:1, :]
            data_2 = psp_origin[..., 2:3, :]
            data_3 = psp_origin[..., 3:4, :]
            data_4 = psp_origin[..., 4:5, :]
            data_6 = psp_origin[..., 6:7, :]
            data_7 = psp_origin[..., 7:8, :]
            split_data[name_type_dataset] = tf.concat([data_0, data_2, data_3, data_4, data_6, data_7], axis=3)

        elif name_type_dataset == 'psp_7':
            start = self.rgb_channels + self.sp_channels + self.uv_channels + self.prgb_channels
            psp_origin = data[..., start:start + self.sp_channels, :]
            data_0 = psp_origin[..., 0:1, :]
            data_2 = psp_origin[..., 2:3, :]
            data_3 = psp_origin[..., 3:4, :]
            data_4 = psp_origin[..., 4:5, :]
            data_5 = psp_origin[..., 5:6, :]
            data_6 = psp_origin[..., 6:7, :]
            data_7 = psp_origin[..., 7:8, :]
            split_data[name_type_dataset] = tf.concat([data_0, data_2, data_3, data_4, data_5, data_6, data_7], axis=3)

        elif '+' in name_type_dataset:
            # Handle combined keys (e.g., 'sp+uv', 'psp+uv')
            sub_keys = name_type_dataset.split('+')
            for d in [self.splitData(data, sub_key) for sub_key in sub_keys]:
                split_data.update(d)
        else:
            raise ValueError(f"Unknown key: {name_type_dataset}")
        return split_data
    
    def splitShape(self, name_type_dataset):
        """
        Splits the data into subsets based on the provided keys.

        Args:
            data: Input tensor of shape (batch_size, height, width, total_channels, 1).
            keys: A list of keys indicating the required data splits.
                  E.g., ['sp', 'psp+uv', 'sp+uv']

        Returns:
            A dictionary where keys are the same as input keys, and values are the corresponding split data.
        """
        input_shapes = {}
        if name_type_dataset == 'rgb':
            input_shapes[name_type_dataset] = (128, 128, 3, 1)
        elif name_type_dataset == 'sp':
            input_shapes[name_type_dataset] = (128, 128, 8, 1)
        elif name_type_dataset == 'sp_5':
            input_shapes[name_type_dataset] = (128, 128, 5, 1)
        elif name_type_dataset == 'sp_3':
            input_shapes[name_type_dataset] = (128, 128, 3, 1)
        elif name_type_dataset == 'sp_4':
            input_shapes[name_type_dataset] = (128, 128, 4, 1)
        elif name_type_dataset == 'sp_6':
            input_shapes[name_type_dataset] = (128, 128, 6, 1)
        elif name_type_dataset == 'sp_7':
            input_shapes[name_type_dataset] = (128, 128, 7, 1)
        elif name_type_dataset == 'uv':
            input_shapes[name_type_dataset] = (128, 128, 1, 1)
        elif name_type_dataset == 'prgb':
            input_shapes[name_type_dataset] = (128, 128, 3, 1)
        elif name_type_dataset == 'psp':
            input_shapes[name_type_dataset] = (128, 128, 8, 1)
        elif name_type_dataset == 'psp_5':
            input_shapes[name_type_dataset] = (128, 128, 5, 1)
        elif name_type_dataset == 'psp_3':
            input_shapes[name_type_dataset] = (128, 128, 3, 1)
        elif name_type_dataset == 'psp_4':
            input_shapes[name_type_dataset] = (128, 128, 4, 1)
        elif name_type_dataset == 'psp_6':
            input_shapes[name_type_dataset] = (128, 128, 6, 1)
        elif name_type_dataset == 'psp_7':
            input_shapes[name_type_dataset] = (128, 128, 7, 1)
        elif '+' in name_type_dataset:
            # Handle combined keys (e.g., 'sp+uv', 'psp+uv')
            sub_keys = name_type_dataset.split('+')
            for d in [self.splitShape(sub_key) for sub_key in sub_keys]:
                input_shapes.update(d)
        else:
            raise ValueError(f"Unknown key: {name_type_dataset}")
        return input_shapes


# Patch Embedding Layer
class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.projection = tf.keras.layers.Dense(embed_dim)

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patch_h, patch_w = self.patch_size

        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, patch_h, patch_w, 1],
            strides=[1, patch_h, patch_w, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dims = tf.shape(patches)[-1]
        num_patches = tf.shape(patches)[1] * tf.shape(patches)[2]
        patches = tf.reshape(patches, [batch_size, num_patches, patch_dims])
        embeddings = self.projection(patches)
        return embeddings

# Positional Embedding (learnable)
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.pos_emb = tf.keras.layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        return x + self.pos_emb(positions)


# Transformer Encoder Block
def transformer_encoder(x, embed_dim, num_heads, ff_dim, dropout=0.1):
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    x = tf.keras.layers.LayerNormalization()(x + attn_output)

    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation='relu'),
        tf.keras.layers.Dense(embed_dim),
        tf.keras.layers.Dropout(dropout),
    ])
    x = tf.keras.layers.LayerNormalization()(x + ffn(x))
    return x

def transformer_encoder_block(x, embed_dim, num_heads, ff_dim, dropout=0.1, num_layers=4):
    for _ in range(num_layers):
        x = PositionalEmbedding(max_len=500, embed_dim=embed_dim)(x)
        x = transformer_encoder(x, embed_dim, num_heads, ff_dim, dropout)
    return x


def create_hybrid_model(num_classes, name_type_dataset, flag_aug, mix_type="con", 
                           patch_size=(4,4), embed_dim=64, num_heads=4, ff_dim=128, dropout=0.1):
    
    input_tensor = layers.Input(shape=(128, 128, 23, 1), name="combined_input")
    inputs = input_tensor
    splitter = DatasetSplitter()
    input_shapes = splitter.splitShape(name_type_dataset)
    split_data = splitter.splitData(inputs, name_type_dataset)

    initializer = tf.keras.initializers.HeNormal(seed=42)

    # Step 1: Patch Embedding and Positional Encoding
    modality_outputs = {}
    for key, shape in input_shapes.items():
        if key not in name_type_dataset:
            continue
        x = split_data[key]
        if key.startswith("sp") or key.startswith("psp"):
            x = tf.squeeze(x, axis=-1)
        if flag_aug:
            x = SimpleAugmentation()(x)
        x = PatchEmbedding(patch_size, embed_dim)(x)
        x = transformer_encoder_block(x, embed_dim, num_heads, ff_dim, dropout, num_layers=4)
        modality_outputs[key] = x

    # Step 2: Residual-style merging
    # merged1 = tf.add_n(list(modality_outputs.values()))
    merged1 = tf.reduce_mean(tf.stack(list(modality_outputs.values()), axis=0), axis=0)

    # Step 3: Split again into modalities and apply another transformer block
    modality_outputs2 = {}
    for key in modality_outputs:
        x = modality_outputs[key]
        x = transformer_encoder_block(layers.LayerNormalization()(x + merged1), embed_dim, num_heads, ff_dim, dropout, num_layers=4)
        modality_outputs2[key] = x

    # Step 4: Merge again
    # merged2 = tf.add_n(list(modality_outputs2.values()))
    merged2 = tf.reduce_mean(tf.stack(list(modality_outputs2.values()), axis=0), axis=0)

    # Step 5: Third transformer stage per modality
    modality_outputs3 = {}
    for key in modality_outputs2:
        x = modality_outputs2[key]
        x = transformer_encoder_block(layers.LayerNormalization()(x + merged2), embed_dim, num_heads, ff_dim, dropout, num_layers=4)
        modality_outputs3[key] = x

    # Step 6: Global Average Pooling & Final Fusion
    pooled_outputs = [layers.GlobalAveragePooling1D()(x) for x in modality_outputs3.values()]
    
    if mix_type == "con":
        combined_features = layers.Concatenate()(pooled_outputs)
    elif mix_type == "mul":
        combined_features = layers.Multiply()(pooled_outputs)
    elif mix_type == "add":
        combined_features = layers.Add()(pooled_outputs)
    else:
        combined_features = layers.Concatenate()(pooled_outputs)

    x = layers.Dense(128, activation='relu', kernel_initializer=initializer)(combined_features)
    x = layers.Dropout(0.1, seed=42)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    return tf.keras.Model(inputs=input_tensor, outputs=outputs)

# model generation function
def cus_ct(num_class=2, name_type_dataset='sp', mix_type="con", flag_aug=False):
    return create_hybrid_model(num_class, name_type_dataset, flag_aug)