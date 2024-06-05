import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='./train_data/code.txt',
    input_format='text',
    model_prefix='code',
    model_type='unigram',
    vocab_size=19000,  # 40000 for japanese
    character_coverage=0.9995,
    input_sentence_size=200000000,
    num_threads=120,
    max_sentencepiece_length=16,
    max_sentence_length=4096,
    split_by_whitespace=True,
    split_digits=True,
    allow_whitespace_only_pieces=True,
    user_defined_symbols=list(set([
        '\n','。','、','.','+','-','/','*','=','(',')','{','}','[',']','^','<',
        '\n','+','-','/','*','=','(',')','{','}','[',']','^','<',
        'auto','break','case','char','const','continue','default','do','double','else','enum','extern','float','for','goto','if','int','long','register','return','signed','sizeof','short','static','struct','switch','typedef','union','unsigned','void','volatile','while',
        'False','None','True','and','as','assert','async','await','break','class','continue','def','del','elif','else','except','finally','for','from','global','if','import','in','is','lambda','nonlocal','not','or','pass','raise','return','try','while','with','yield',
        'alignas','alignof','and','and_eq','asm','atomic_cancel','atomic_commit','atomic_noexcept','auto','bitand','bitor','bool','break','case','catch',   'char','char8_t','char16_t','char32_t','class','compl','concept','const','consteval','constexpr','constinit','const_cast','continue','co_await','co_return','co_yield','decltype','default','delete','do','double','dynamic_cast','else','enum','explicit','export','extern','false','float','for','friend','goto','if','import','inline','int','long','mutable','namespace','new','noexcept','not','not_eq','nullptr','operator','or','or_eq','private','protected','public','reflexpr','register','reinterpret_cast','requires','return','short','signed','sizeof','static','static_assert','static_cast','struct','switch','synchronized','template','this','thread_local','throw','true','try','typedef','typeid','typename','union','unsigned','using','virtual','void','volatile','wchar_t','while','xor','xor_eq',
        '_START_ARTICLE_', '_START_SECTION_', '_START_PARAGRAPH_', '_NEWLINE_'
    ])),
    byte_fallback=True,
    normalization_rule_name='nmt_nfkc',
    remove_extra_whitespaces=False,
    unk_piece='<unk>',
    bos_piece='<s>',
    eos_piece='</s>',
    pad_piece='<pad>',
    train_extremely_large_corpus=True,
)
