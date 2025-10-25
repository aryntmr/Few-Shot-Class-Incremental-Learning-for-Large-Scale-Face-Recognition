from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops

class DaliPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data, labels, k):
        super(DaliPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.ExternalSource()
        self.input_labels = ops.ExternalSource()
        self.decode = ops.ImageDecoder(output_type=types.RGB)
        self.resize = ops.Resize(device="gpu", resize_shorter=k)
        self.data = data
        self.labels = labels

    def define_graph(self):
        images = self.decode(self.input())
        labels = self.input_labels()
        images = self.resize(images)
        return images, labels

    def iter_setup(self):
        # Ensure that self.data and self.labels are DataNode objects
#        data_node = fn.external_source(source=self.data, num_outputs=1)
#        labels_node = fn.external_source(source=self.labels, num_outputs=1)
        # Feed DALI tensors into the pipeline
#        self.feed_input(data_node, labels_node)
        '''
        # Define a function to feed data into the pipeline
        def feed_data():
            for d in self.data:
                yield d

        def feed_labels():
            for l in self.labels:
                yield l

        # Feed DALI tensors into the pipeline
        self.feed_input(self.input, feed_data)
        self.feed_input(self.input_labels, feed_labels)
        '''
        pass

    def define_inputs(self, pipeline, batch_size):
        self.data_src = pipeline.set_external_source(source=self.data, layout="HWC", name="data")
        self.labels_src = pipeline.set_external_source(source=self.labels, layout="C", name="labels")

class ExternalSourceWrapper:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        self.i = 0
        self.n = len(self.data)
        return self

    def __next__(self):
        if self.i < self.n:
            result = self.data[self.i]
            self.i += 1
            return (result,)
        else:
            raise StopIteration

