class QueryIndex:
    def query(self, inputs, mode="all", return_metadata=False):
        """Return matching entries for each input."""
        raise NotImplementedError

    def query_first(self, inputs):
        """Return the first matching entry for each input."""
        raise NotImplementedError
