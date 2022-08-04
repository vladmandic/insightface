from tensorflow.keras.models import Model, load_model
model = load_model("TT_ghostnet_strides_1_prelu_25_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_basic_agedb_30_epoch_47_0.980000.h5", compile=False)
model.save("insightface-ghostnet-strides1")
