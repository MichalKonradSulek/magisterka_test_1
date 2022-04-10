class TrainedNet:
    def __init__(self, encoder, decoder, feed_height, feed_width, device):
        self.encoder = encoder
        self.decoder = decoder
        self.feed_height = feed_height
        self.feed_width = feed_width
        self.device = device
