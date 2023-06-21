# TODO implement
# import stellargraph as sg
# from stellargraph.mapper import GraphSAGENodeGenerator
# from stellargraph.layer import GraphSAGE


# generator = GraphSAGENodeGenerator(
#     graph,
#     batch_size=32,
#     num_samples=[5, 2]
#     )

# sage_model = GraphSAGE(
#     layer_sizes=[128, 128],
#     generator=generator,
#     bias=True,
#     dropout=0.5
#     )

# x_inp, x_out = sage_model.build()

# model = tf.keras.Model(inputs=x_inp, outputs=x_out)
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])

# train_gen = generator.flow(train_nodes)
# val_gen = generator.flow(val_nodes)
# history = model.fit(train_gen, epochs=50, validation_data=val_gen)

# all_nodes = graph.nodes()
# node_gen = generator.flow(all_nodes)
# node_embeddings = model.predict(node_gen)
