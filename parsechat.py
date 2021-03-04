import os
import sys
import argparse
import re
from pprint import pprint as pp

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

_url_regex = r'''(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))'''

_regex_token = \
   [(_url_regex, ' <LINK> '),
    ('[\U0001f000-\U0001ffff]', ''),
    (r'\n', ' '),
    (r'\w\. ', ' \. '),
    (r'\.{2,}', ' <...> '),
    (r'\,', ' , '),
    (r'\?+', ' ? '),
    (r'\!+', ' ! '),
    (r' q ', ' que '),
    (r'(k{3,})', ' <LAUGH> '), 
    (r'(h[aeio(uh]+){2,}', ' <LAUGH> '), 
    (r'([aeiou]+h){2,}', ' <LAUGH> '),
    (r'<media omitted>', ' <MEDIA> '),
    (r' hmm*', ' hmm '),
    (r' nã*o* ', ' não '),
    (r' mu+ito ', ' muito '),
    (r' junto+ ', ' junto '),
    (r' dissae+ ', ' disso aí '),
    (r' issae+ ', ' isso aí '),
    (r' bo+a+ ', ' boa '),
    (r' e+h+ ', ' é '),
    (r' ja+h+ ', ' já '),
    (r' fo+i+ ', ' foi '),
    (r' qu[eé]+h* ', ' quer '),
    (r' vc ', ' você '),
    (r' vcs ', ' vocês '),
    (r' algu[ée]+m+ ', ' alguém '),
    (r' judio+ ', ' judio '),
    (r' va+i+ ', ' vai '),
    (r' t[aá]+ ', ' tá '),
    (r' tb+ ', ' também '),
    (r' po+[hr]+a+ ', ' porra '),
    (r' muié ', ' mulher '),
    (r' msmo* ', ' mesmo '),
    (r'  +', ' ')] 


def _replace_tokens(s, _regex_token): 
    x = s.lower()
    for p, t in _regex_token:
        x = re.sub(p, ' ' + t, x) 
    return x

def _msgs_gen(raw_lines):
    '''
    Strips date and time from the start of each message, and concatenates 
    newlines into a single message.

    '''
    date_exp = r'([0-3]?[\d]/){2}[\d][\d], [0-2][\d]:[\d]{2} - '
    pattern = re.compile(date_exp)
    
    msg = ''
    for line in raw_lines:
        match = pattern.match(line)
        if match:
            if msg != '':
                yield msg
            msg = line[match.end():]
        else:
            msg = ''.join([msg, line])


def _user_and_text_gen(msgs, max_text_len=128):
    """
    msgs: sentence with pattern 'r[\w ]+: ' to be parsed to user and text
    max_text_len: maximum length after pos trimming of msg text

    returns: (user, text)
    """
    msg_exp = r'[\w ]+: '
    pattern = re.compile(msg_exp)

    for msg in msgs:
        match = pattern.match(msg)
        if match:
            user = "".join(['<', match[0], '>'])
            user = user.replace(': ', '').replace(' ', '_')
            text = msg[match.end():]
        yield (user, text[:max_text_len])


def _parse_file(raw_lines):
    """
    returns: an iter for parsed and filtered messages.
    """
    msgs_iter = _msgs_gen(raw_lines)
    user_text_iter = _user_and_text_gen(msgs_iter)
    # user, text
    for u, t in user_text_iter: 
        text = _replace_tokens(t, _regex_token)
        if text != ' <MEDIA> ':
            yield ' '.join([u, text])


def parse_tape(dataset_path):
    with open(dataset_path, 'r') as f:
        next(f)
        parsed_file = _parse_file(f)
        msgs_tape = ' <EOM> '.join(parsed_file) 
    return msgs_tape


def export_tokenizer(filepath, dataset_path, max_vocab_size):
    msgs_tape = parse_tape(dataset_path)
    tokenizer = Tokenizer(max_vocab_size, filters='')
    tokenizer.fit_on_texts([msgs_tape])
    with open(filepath, 'w') as f:
        f.write(tokenizer.to_json())


def create_indexes_tape(dataset_path, tokenizer):
    msgs_tape = parse_tape(dataset_path)
    return tokenizer.texts_to_sequences([msgs_tape])[0]

def create_datasets(tape,
                    train_n_batches,
                    val_n_batches,
                    batch_size,
                    seq_len):

    dataset = tf.data.Dataset.from_tensor_slices(tape)
    dataset = dataset.window(seq_len, 1, drop_remainder=True)
    dataset = dataset.flat_map(lambda x: x.batch(seq_len))
    dataset = dataset.map(lambda x: (x[:-1], x[-1]))
    dataset = dataset.shuffle((len(tape)-seq_len), seed=1)
    dataset = dataset.batch(batch_size)
    train_dataset = dataset.take(train_n_batches)
    dataset = dataset.skip(train_n_batches)
    val_dataset = dataset.take(val_n_batches)
    train_dataset = train_dataset.repeat()
    
    return train_dataset, val_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Create tokenizer and save it as json.')
    parser.add_argument('path', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('vocab_size', type=int)
    args = parser.parse_args()
    
    export_tokenizer(args.path, args.dataset, args.vocab_size)

