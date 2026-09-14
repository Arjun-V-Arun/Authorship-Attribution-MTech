==============================================================================
MORPHEME EXPERIMENT — fiction
==============================================================================
114 works, 12 authors

------------------------------------------------------------------------------
1. VOCABULARY — does segmentation collapse the type inventory?
------------------------------------------------------------------------------
  version                      types   top100   top300    top1k   hapax    beta
  raw (whitespace)           151,907   18.1%   26.0%   36.6%    77%   0.865
  punctuation stripped       131,892   19.8%   28.7%   40.4%    75%   0.846
  subword vocab=2,000          2,329   41.3%   62.0%   87.4%    11%   0.252
  subword vocab=8,000          8,329   27.2%   41.7%   61.6%     3%   0.434
  subword vocab=32,000        32,316   20.9%   32.2%   47.6%     1%   0.639

  punctuation alone accounts for 13.2% of the raw type count
  segmentation cuts types by 75.5% and lifts top-300 coverage 28.7% -> 32.2%

------------------------------------------------------------------------------
2. DOES SEGMENTATION REPAIR BURROWS' DELTA?
------------------------------------------------------------------------------
  If inflectional fragmentation is what breaks Delta, this is where
  it shows. No improvement means the explanation is wrong.

  Delta (300 MFW), raw words         macro-F1=0.735 (+/-0.011)
  Delta (300), subword v=2,000       macro-F1=0.853 (+/-0.018)
  Delta (300), subword v=8,000       macro-F1=0.867 (+/-0.010)
  Delta (300), subword v=32,000      macro-F1=0.844 (+/-0.027)
  Delta (1000), raw words            macro-F1=0.635 (+/-0.005)

  >>> DELTA, raw -> segmented: 0.735 -> 0.867 (+0.132)
      Segmentation repairs Delta. The fragmentation explanation is
      supported causally, not just by correlation. This is the
      result that turns the coverage statistics into a mechanism.

------------------------------------------------------------------------------
3. ARE MORPHEMES A USEFUL FEATURE FOR ATTRIBUTION?
------------------------------------------------------------------------------
  char 3-5 gram, raw (reference)     macro-F1=0.904 (+/-0.012)
  word 1-2 gram, raw                 macro-F1=0.913 (+/-0.015)
  subword 1-2 gram, v=2,000          macro-F1=0.886 (+/-0.003)
  subword 1-2 gram, v=8,000          macro-F1=0.912 (+/-0.014)
  subword 1-2 gram, v=32,000         macro-F1=0.936 (+/-0.010)
  subword 1-3 gram, v=2,000          macro-F1=0.935 (+/-0.010)

==============================================================================
MORPHEME EXPERIMENT — essays
==============================================================================
599 works, 16 authors

------------------------------------------------------------------------------
1. VOCABULARY — does segmentation collapse the type inventory?
------------------------------------------------------------------------------
  version                      types   top100   top300    top1k   hapax    beta
  raw (whitespace)           253,334   13.4%   19.6%   28.1%    77%   0.890
  punctuation stripped       223,211   14.4%   21.4%   30.9%    75%   0.874
  subword vocab=2,000          2,084   44.2%   63.7%   88.3%     1%   0.237
  subword vocab=8,000          8,084   28.0%   41.7%   60.5%     0%   0.402
  subword vocab=32,000        32,066   20.7%   30.9%   45.1%     0%   0.601

  punctuation alone accounts for 11.9% of the raw type count
  segmentation cuts types by 85.6% and lifts top-300 coverage 21.4% -> 30.9%

------------------------------------------------------------------------------
2. DOES SEGMENTATION REPAIR BURROWS' DELTA?
------------------------------------------------------------------------------
  If inflectional fragmentation is what breaks Delta, this is where
  it shows. No improvement means the explanation is wrong.

  Delta (300 MFW), raw words         macro-F1=0.678 (+/-0.001)
  Delta (300), subword v=2,000       macro-F1=0.775 (+/-0.009)
  Delta (300), subword v=8,000       macro-F1=0.793 (+/-0.010)
  Delta (300), subword v=32,000      macro-F1=0.764 (+/-0.016)
  Delta (1000), raw words            macro-F1=0.633 (+/-0.006)

  >>> DELTA, raw -> segmented: 0.678 -> 0.793 (+0.114)
      Segmentation repairs Delta. The fragmentation explanation is
      supported causally, not just by correlation. This is the
      result that turns the coverage statistics into a mechanism.

------------------------------------------------------------------------------
3. ARE MORPHEMES A USEFUL FEATURE FOR ATTRIBUTION?
------------------------------------------------------------------------------
  char 3-5 gram, raw (reference)     macro-F1=0.914 (+/-0.010)
  word 1-2 gram, raw                 macro-F1=0.905 (+/-0.002)
  subword 1-2 gram, v=2,000          macro-F1=0.907 (+/-0.003)
  subword 1-2 gram, v=8,000          macro-F1=0.908 (+/-0.001)
  subword 1-2 gram, v=32,000         macro-F1=0.922 (+/-0.018)
  subword 1-3 gram, v=2,000          macro-F1=0.909 (+/-0.001)

