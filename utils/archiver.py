import os
import zstandard
import json
import jsonlines
import io
import datetime


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime.datetime,)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


# Modified version of lm_dataformat Archive for single file.
class Archive:
    def __init__(self, file_path, compression_level=3):
        self.file_path = file_path
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)    
        self.fh = open(self.file_path, 'wb')
        self.cctx = zstandard.ZstdCompressor(level=compression_level)
        self.compressor = self.cctx.stream_writer(self.fh)
    
    def add_data(self, data, meta={}):
        self.compressor.write(json.dumps({'text': data, 'meta': meta}, default=json_serial).encode('UTF-8') + b'\n')
    
    def commit(self):
        self.compressor.flush(zstandard.FLUSH_FRAME)        
        self.fh.flush()
        self.fh.close()


# Modified version of lm_dataformat Reader with self.fh set, allowing peeking for tqdm.
class Reader:
    def __init__(self):
        pass

    def read_jsonl(self, file, get_meta=False, autojoin_paragraphs=True, para_joiner='\n\n'):
        with open(file, 'rb') as fh:
            self.fh = fh
            cctx = zstandard.ZstdDecompressor()
            reader = io.BufferedReader(cctx.stream_reader(fh))
            rdr = jsonlines.Reader(reader)
            for ob in rdr:
                # naive jsonl where each object is just the string itself, with no meta. For legacy compatibility.
                if isinstance(ob, str):
                    assert not get_meta
                    yield ob
                    continue

                text = ob['text'] if 'text' in ob else ob['content']
                if autojoin_paragraphs and isinstance(text, list):
                    text = para_joiner.join(text)

                if get_meta:
                    yield text, (ob['meta'] if 'meta' in ob else {})
                else:
                    yield text

    def read_jsonl2(self, file, get_meta=False, get_all=False):
        with jsonlines.open(file, mode='r') as fr:
            for ob in fr:
                if get_all:
                    yield ob
                else:
                    text = ob['text'] if 'text' in ob else ob['content']
                    if get_meta:
                        yield text, ob['meta'] if 'meta' in ob else {}
                    else:
                        yield text
