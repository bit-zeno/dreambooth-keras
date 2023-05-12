import tensorflow as tf
from tensorflow.keras import layers, Model

def get_diffusion_sub_models(diffusion_model):
    dm = diffusion_model

    m0_i1 = dm.layers[0].input
    m0_i2 = dm.layers[2].input

    m0 = dm.layers[1](m0_i1)
    m0 = dm.layers[3](m0)
    m0_1 = dm.layers[4](m0_i2)
    m0_2 = dm.layers[5](m0) #dense_1
    m0 = dm.layers[6]([m0_1,m0_2])

    rb0 = Model([m0_i1,m0_i2], [m0,m0_1,m0_2])


    emb_i1 = dm.layers[7].input
    ip_m0 = layers.Input(shape=(64, 64, 320)) #same as m1 output
    ip_m0_2 = layers.Input(shape=(1280)) #same as m1_2 outpout

    m1_1 = dm.layers[8]([ip_m0,emb_i1])
    m1 = dm.layers[9]([m1_1,ip_m0_2])

    rb1 = Model([ip_m0,emb_i1,ip_m0_2], [m1,m1_1])


    ip_m1 = layers.Input(shape=(64, 64, 320)) #same as m1 output

    m2_1 = dm.layers[10]([ip_m1,emb_i1])
    m2_2 = dm.layers[11](m2_1)
    m2 = dm.layers[12]([m2_2,ip_m0_2])

    rb2 = Model([ip_m1,ip_m0_2,emb_i1], [m2,m2_1,m2_2])


    ip_m2 = layers.Input(shape=(32, 32, 640)) #same as m2 output

    m3_1 = dm.layers[13]([ip_m2,emb_i1])
    m3 = dm.layers[14]([m3_1,ip_m0_2])

    rb3 = Model([ip_m2,ip_m0_2,emb_i1], [m3,m3_1])


    ip_m3 = layers.Input(shape=(32, 32, 640)) #same as m3 output

    m4_1 = dm.layers[15]([ip_m3,emb_i1])
    m4_2 = dm.layers[16](m4_1)
    m4 = dm.layers[17]([m4_2,ip_m0_2])

    rb4 = Model([ip_m3,ip_m0_2,emb_i1], [m4,m4_1,m4_2])


    ip_m4 = layers.Input(shape=(16, 16, 1280)) #same as m4 output

    m5_1 = dm.layers[18]([ip_m4,emb_i1])
    m5 = dm.layers[19]([m5_1,ip_m0_2])

    rb5 = Model([ip_m4,ip_m0_2,emb_i1], [m5,m5_1])


    ip_m5 = layers.Input(shape=(16, 16, 1280)) #same as m5 output

    m6_1 = dm.layers[20]([ip_m5,emb_i1])
    m6_2 = dm.layers[21](m6_1)
    m6 = dm.layers[22]([m6_2,ip_m0_2])

    rb6 = Model([ip_m5,ip_m0_2,emb_i1], [m6,m6_1,m6_2])


    ip_m6 = layers.Input(shape=(8, 8, 1280)) #same as m6 output

    m7 = dm.layers[23]([ip_m6,ip_m0_2])
    m8 = dm.layers[24]([m7,ip_m0_2])
    m9 = dm.layers[25]([m8,emb_i1])
    m9 = dm.layers[26]([m9,ip_m0_2])

    rb9 = Model([ip_m6,ip_m0_2,emb_i1], [m7,m9])


    ip_m7 = layers.Input(shape=(8, 8, 1280)) #same as m7 output
    ip_m9 = layers.Input(shape=(8, 8, 1280)) #same as m9 output

    m10 = dm.layers[27]([ip_m9,ip_m7])
    m10 = dm.layers[28]([m10,ip_m0_2])

    rb10 = Model([ip_m7,ip_m9,ip_m0_2], m10)


    ip_m10 = layers.Input(shape=(8, 8, 1280)) #same as m10 output

    m11 = dm.layers[29]([ip_m10,ip_m6])
    m11 = dm.layers[30]([m11,ip_m0_2])

    rb11 = Model([ip_m6,ip_m10,ip_m0_2], m11)


    ip_m11 = layers.Input(shape=(8, 8, 1280)) #same as m11 output
    ip_m6_2 = layers.Input(shape=(8, 8, 1280)) #same as m6_2 (padded_conv2d_27) output

    m12 = dm.layers[31]([ip_m11,ip_m6_2])
    m12 = dm.layers[32]([m12,ip_m0_2])

    rb12 = Model([ip_m11,ip_m6_2,ip_m0_2], m12)


    ip_m12 = layers.Input(shape=(8, 8, 1280))
    ip_m6_1 = layers.Input(shape=(16, 16, 1280)) #same as m6_2 (spatial_transformer_5) output

    m13 = dm.layers[33](ip_m12)
    m13 = dm.layers[34]([m13,ip_m6_1])
    m13 = dm.layers[35]([m13,ip_m0_2])

    rb13 = Model([ip_m12,ip_m6_1,ip_m0_2], m13)


    ip_m13 = layers.Input(shape=(16, 16, 1280))
    ip_m5_1 = layers.Input(shape=(16, 16, 1280)) #same as m5_1 (spatial_transformer_4) output

    m14 = dm.layers[36]([ip_m13,emb_i1])
    m14 = dm.layers[37]([m14,ip_m5_1])
    m14 = dm.layers[38]([m14,ip_m0_2])

    rb14 = Model([ip_m13,ip_m5_1,emb_i1,ip_m0_2], m14)


    ip_m14 = layers.Input(shape=(16, 16, 1280))
    ip_m4_2 = layers.Input(shape=(16, 16, 640)) #same as m4_2 (padding_conv2d_18) output

    m15 = dm.layers[39]([ip_m14,emb_i1])
    m15 = dm.layers[40]([m15,ip_m4_2])
    m15 = dm.layers[41]([m15,ip_m0_2])

    rb15 = Model([ip_m14,ip_m4_2,emb_i1,ip_m0_2], m15)


    ip_m15 = layers.Input(shape=(16, 16, 1280))
    ip_m4_1 = layers.Input(shape=(32, 32, 640)) #same as m4_1 (spatial_transformer_3) output

    m16 = dm.layers[42]([ip_m15,emb_i1])
    m16 = dm.layers[43](m16)
    m16 = dm.layers[44]([m16,ip_m4_1])
    m16 = dm.layers[45]([m16,ip_m0_2])

    rb16 = Model([ip_m15,ip_m4_1,emb_i1,ip_m0_2], m16)


    ip_m16 = layers.Input(shape=(32, 32, 640))
    ip_m3_1 = layers.Input(shape=(32, 32, 640))

    m17 = dm.layers[46]([ip_m16,emb_i1])
    m17 = dm.layers[47]([m17,ip_m3_1])
    m17 = dm.layers[48]([m17,ip_m0_2])

    rb17 = Model([ip_m16,ip_m3_1,emb_i1,ip_m0_2], m17)


    ip_m17 = layers.Input(shape=(32, 32, 640))
    ip_m2_2 = layers.Input(shape=(32, 32, 320))

    m18 = dm.layers[49]([ip_m17,emb_i1])
    m18 = dm.layers[50]([m18,ip_m2_2])
    m18 = dm.layers[51]([m18,ip_m0_2])

    rb18 = Model([ip_m17,ip_m2_2,emb_i1,ip_m0_2], m18)


    ip_m18 = layers.Input(shape=(32, 32, 640))
    ip_m2_1 = layers.Input(shape=(64, 64, 320))

    m19 = dm.layers[52]([ip_m18,emb_i1])
    m19 = dm.layers[53](m19)
    m19 = dm.layers[54]([m19,ip_m2_1])
    m19 = dm.layers[55]([m19,ip_m0_2])

    rb19 = Model([ip_m18,ip_m2_1,emb_i1,ip_m0_2], m19)


    ip_m19 = layers.Input(shape=(64, 64, 320))
    ip_m1_1 = layers.Input(shape=(64, 64, 320))

    m20 = dm.layers[56]([ip_m19,emb_i1])
    m20 = dm.layers[57]([m20,ip_m1_1])
    m20 = dm.layers[58]([m20,ip_m0_2])

    rb20 = Model([ip_m19,ip_m1_1,emb_i1,ip_m0_2], m20)


    ip_m20 = layers.Input(shape=(64, 64, 320))
    ip_m0_1 = layers.Input(shape=(64, 64, 320))

    m21 = dm.layers[59]([ip_m20,emb_i1])
    m21 = dm.layers[60]([m21,ip_m0_1])
    m21 = dm.layers[61]([m21,ip_m0_2])

    rb21 = Model([ip_m20,ip_m0_1,emb_i1,ip_m0_2], m21)


    ip_m21 = layers.Input(shape=(64, 64, 320))

    mfinal = dm.layers[62]([ip_m21,emb_i1])
    mfinal = dm.layers[63](mfinal)
    mfinal = dm.layers[64](mfinal)
    mfinal = dm.layers[65](mfinal)

    mm = Model([ip_m21,emb_i1], mfinal)

    return [rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb9, rb10, rb11, rb12, rb13, rb14, rb15, rb16, rb17, rb18, rb19, rb20, rb21, mm]

def get_recompute_models(submodels: list[Model]):
    return [tf.recompute_grad(sm) for sm in submodels]

@tf.function
def forward_pass_with_recompute(recompute_models, t0_i1, t0_i2, embt_i1):
    rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb9, rb10, rb11, rb12, rb13, rb14, rb15, rb16, rb17, rb18, rb19, rb20, rb21, mm = recompute_models

    o0, o0_1, o0_2 = rb0([t0_i1,t0_i2])
    o1, o1_1 = rb1([o0,embt_i1,o0_2])
    o2, o2_1, o2_2 = rb2([o1,o0_2,embt_i1])
    o3, o3_1 = rb3([o2,o0_2,embt_i1])
    o4, o4_1, o4_2 = rb4([o3,o0_2,embt_i1])
    o5, o5_1 = rb5([o4,o0_2,embt_i1])
    o6, o6_1, o6_2 = rb6([o5,o0_2,embt_i1])
    o7, o9 = rb9([o6,o0_2,embt_i1])
    o10 = rb10([o7,o9,o0_2])
    o11 = rb11([o6,o10,o0_2])
    o12 = rb12([o11,o6_2,o0_2])
    o13 = rb13([o12,o6_1,o0_2])
    o14 = rb14([o13,o5_1,embt_i1,o0_2])
    o15 = rb15([o14,o4_2,embt_i1,o0_2])
    o16 = rb16([o15,o4_1,embt_i1,o0_2])
    o17 = rb17([o16,o3_1,embt_i1,o0_2])
    o18 = rb18([o17,o2_2,embt_i1,o0_2])
    o19 = rb19([o18,o2_1,embt_i1,o0_2])
    o20 = rb20([o19,o1_1,embt_i1,o0_2])
    o21 = rb21([o20,o0_1,embt_i1,o0_2])
    omm = mm([o21,embt_i1])

    return omm