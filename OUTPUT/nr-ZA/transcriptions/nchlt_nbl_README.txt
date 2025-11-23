--------------------------------------------------------------------------------
README: NCHLT Speech Corpus
--------------------------------------------------------------------------------


Full name:        NCHLT Speech Corpus


Description: Orthographically transcribed broadband speech corpora for all of 
South Africa’s eleven official languages. See "Detailed information" for more 
information.


Languages:        
Afrikaans                 (afr)
English                   (eng)
isiNdebele                (nbl)
isiXhosa                  (xho)
isiZulu                   (zul)
Setswana                  (tsn)
Sesotho sa Leboa          (nso)
Sesotho                   (sot)
siSwati                   (ssw)
Tshivenda                 (ven)
Xitsonga                  (tso)


Version:      0.1


Size:         50.6GB (~4.6GB per language)


URL:          http://rma.nwu.ac.za/
              Any queries with regard to updated versions can be directed to
              rma@nwu.ac.za.


Subsidiary
resources:    https://sites.google.com/site/nchltspeechcorpus


Note:        This corpus is also referred to as nchlt-clean (see [1,2]) in order
             to contrast it with two other corpora: 
             - nchlt-raw (all utterances collected, including problematic
               utterances) and 
             - nchlt-baseline (no duplicate speakers).
             For more information see [1,2] and subsidiary resources.

--------------------------------------------------------------------------------


This data is shared under the Creative Commons Attribution 3.0 Unported 
(CC BY 3.0) license. For more information see LICENSE.txt


When using this corpus, please cite:
  Etienne Barnard, Marelie H. Davel, Charl van Heerden, Febe de Wet and Jaco 
  Badenhorst, "The NCHLT Speech Corpus of the South African languages," In Proc. 
  4th International Workshop on Spoken Language Technologies for Under-resourced 
  Languages (SLTU), St Petersburg, Russia, May 2014.


bibtex:
@inproceedings{barnard14nchltspeechcorpus,
  author   = {Etienne Barnard, Marelie H. Davel, Charl van Heerden, Febe de Wet
               and Jaco Badenhorst},
  title     = {{T}he {NCHLT} {S}peech {C}orpus of the {S}outh {A}frican 
               languages},
  booktitle = {Proc. 4th International Workshop on Spoken Language Technologies 
               for Under-resourced Languages (SLTU)},
  address   = {St Petersburg, Russia},
  year      = {2014},
  month     = {May}
}

--------------------------------------------------------------------------------

DETAILED INFORMATION


DESCRIPTION:


The complete orthographically transcribed broadband speech corpora for all of 
South Africa’s eleven official languages are given i.t.o. an audio and 
transcriptions directory on a per language basis. Transcriptions are given in 
XML format and subdivided into individual train and test suites 
("nchlt_<language_code>.trn.xml" and "nchlt_<language_code>.tst.xml" respectively). 
With each XML entry in these files, additional metadata is provided (a) per speaker:
  - recording location
  - age
  - gender
and (b) per file:
  - wav file md5sum
  - wav file duration (seconds)
  - pdp_score (see [4] for more detail)

In cases where the metadata failed basic checks, e.g. invalid ID numbers, 
or was not available, the corresponding field contains the value "-1".
The majority of the speakers are in the age range between 18-55 and the ratio
between male and female speakers is close to 50:50 for each language.

Individual recordings are provided using the WAVE format (16-bit, mono,
PCM sampled at 16kHz) and is subdivided using a unique speaker identifier
(<spk_id>) for every speaker. Speaker identifiers for the test suite audio
correspond to the integer values of 500-507.

--------------------------------------------------------------------------------


CORPUS DIRECTORY/FILE STRUCTURE:


nchlt_<ISO 639-3>
├── audio
│   ├── <spk_id>
│   │   ├── nchlt_<ISO 639-3>_<spk_id><gender>_<file_number>.wav
│   ...
└── transcriptions
   ├── nchlt_<ISO 639-3>.trn.xml
   └── nchlt_<ISO 639-3>.tst.xml

--------------------------------------------------------------------------------


ADDITIONAL DOCUMENTATION:


[1]        Etienne Barnard, Marelie H. Davel, Charl van Heerden, Febe de Wet and 
Jaco Badenhorst, "The NCHLT Speech Corpus of the South African 
languages," In Proc. 4th International Workshop on Spoken Language 
Technologies for Under-resourced Languages (SLTU), St Petersburg, 
Russia, May 2014.


[2]        Charl van Heerden, Marelie H. Davel and Etienne Barnard, "The 
semi-automated creation of stratiﬁed speech corpora," In Proc. Pattern 
Recognition Association of South Africa annual symposium (PRASA), 
Johannesburg, South Africa, Dec 2013, pp 115-119.


[3]        N.J. de Vries, M.H. Davel, J. Badenhorst, W.D. Basson, F. de Wet, E. 
Barnard and A. de Waal, "A smartphone-based ASR data collection tool for 
under-resourced languages," Speech Communication, Volume 56, January 
2014, pp 119–131.


[4]        Marelie H. Davel, Charl van Heerden, and Etienne Barnard, "Validating 
Smartphone-Collected Speech Corpora," in In Proc. 3rd International 
Workshop on Spoken Language Technologies for Under-resourced Languages 
(SLTU), Cape Town, South Africa, May 2012, pp. 68–75.


[5]        C van Heerden, M.H. Davel and E. Barnard, "Medium-Vocabulary Speech 
Recognition for Under-Resourced Languages", in In Proc. 3rd 
International Workshop on Spoken Language Technologies for 
Under-resourced Languages (SLTU), Cape Town, South Africa, May 2012, pp. 
146-151.


[6]        J. Badenhorst, A. De Waal  and F. de Wet, "Quality measurements for  
mobile data collection in the developing world", in In Proc. 3rd 
International Workshop on Spoken Language Technologies for 
Under-resourced Languages (SLTU), Cape Town, South Africa, May 2012, pp. 
139-145.


--------------------------------------------------------------------------------
CSIR Meraka Human Language Technologies
--------------------------------------------------------------------------------