--------------------------------------------------------------------------------
README: nchltAux1 Speech Corpus
--------------------------------------------------------------------------------

Full name:        nchltAux1 Speech Corpus

Description: Orthographically transcribed broadband speech in South Africa’s 
eleven official languages. See "Detailed Information" for more information.

Languages:              ISO 639-3        Size          Duration    # of Speakers
-------------------------------------------------------------------------------------        
Afrikaans                 afr            4.8 G         42:57:48       210
English                   eng            3.4 G         30:00:34       210
isiNdebele                nbl            4.7 G         42:45:06       148
Sesotho sa Leboa          nso            7.2 G         65:14:39       210
Sesotho                   sot            8.2 G         74:14:39       210
siSwati                   ssw            8.7 G         78:48:56       197
Setswana                  tsn            7.8 G         70:31:43       210
Xitsonga                  tso            9.2 G         84:02:27       198
Tshivenda                 ven           11.0 G         94:07:16       208 
isiXhosa                  xho           12.0 G        103:24:14       209
isiZulu                   zul           11.0 G         98:20:26       210

Version:      0.1

Size:         88 G
Duration:     784:27:48 (Hours:Minutes:Seconds)
Speakers:     2 220

--------------------------------------------------------------------------------

This data is shared under the Creative Commons Attribution 3.0 Unported 
(CC BY 3.0) license. For more information see LICENSE.txt

When using this corpus, please cite:

  Jaco Badenhorst, Laura Martinus and Febe de Wet, "BLSTM harvesting of auxiliary NCHLT speech data," 
  In Proceedings of SAUPEC/ROBMECH/PRASA 2019, Bloemfontein, South Africa, January 2019.

bibtex:
@inproceedings{badenhorst19nchltaux,
  author   = {Jaco Badenhorst, Laura Martinus and Febe de Wet},
  title     = {{BLSTM} harvesting of auxiliary {NCHLT} speech data},
  booktitle = {Proceedings of SAUPEC/ROBMECH/PRASA},
  address   = {Bloemfontein, South Africa},
  year      = {2019},
  month     = {January}
}

--------------------------------------------------------------------------------

DETAILED INFORMATION

The corpus contains orthographically transcribed broadband speech in each of 
South Africa’s eleven official languages. Transcriptions are provided in 
XML format. 

The following meta-data is provided for each entry in the XML files:
(a) per speaker:
  - recording location
  - age
  - gender
(b) per file:
  - wav file md5sum
  - wav file duration (seconds)
  - pdp_score (see [1] and [5] for more detail)

In cases where the metadata failed basic checks (e.g. invalid ID numbers) 
or was not available, the corresponding field contains the value "-1".
The majority of the speakers are in the age range between 18-55 and the ratio
between male and female speakers is close to 50:50 for each language.

Individual recordings are provided in WAVE format (16-bit, mono,PCM sampled at 16kHz) 
and is subdivided using a unique speaker identifier (<spk_id>) for every speaker. 

--------------------------------------------------------------------------------

CORPUS DIRECTORY/FILE STRUCTURE:

<ISO 639-3 lang code>
├── audio
│   ├── <spk_id>
│   │   ├── nchltAux1_<ISO 639-3 lang code>_<spk_id><gender>_<file_number>.wav
│   ...
└── info
   ├── nchltAux1_<ISO 639-3 lang code>.xml            (pdp scores derived as in [1])
   └── nchltAux1_<ISO 639-3 lang code>_old_scores.xml (pdp scores derived as in [5])
   └── nchlt_<ISO 639-3 lang code>_aux.dict                (pronunciations for transcriptions)

--------------------------------------------------------------------------------

ADDITIONAL DOCUMENTATION:

[1]        Jaco Badenhorst, Laura Martinus and Febe de Wet, "BLSTM harvesting of auxiliary NCHLT 
speech data," In Proceedings of SAUPEC/ROBMECH/PRASA 2019, Bloemfontein, 
South Africa, January 2019.

[2]        Etienne Barnard, Marelie H. Davel, Charl van Heerden, Febe de Wet and 
Jaco Badenhorst, "The NCHLT Speech Corpus of the South African 
languages," In Proc. 4th International Workshop on Spoken Language 
Technologies for Under-resourced Languages (SLTU), St Petersburg, 
Russia, May 2014.

[3]        Charl van Heerden, Marelie H. Davel and Etienne Barnard, "The 
semi-automated creation of stratiﬁed speech corpora," In Proc. Pattern 
Recognition Association of South Africa annual symposium (PRASA), 
Johannesburg, South Africa, Dec 2013, pp 115-119.

[4]        N.J. de Vries, M.H. Davel, J. Badenhorst, W.D. Basson, F. de Wet, E. 
Barnard and A. de Waal, "A smartphone-based ASR data collection tool for 
under-resourced languages," Speech Communication, Volume 56, January 
2014, pp 119–131.

[5]        Marelie H. Davel, Charl van Heerden, and Etienne Barnard, "Validating 
Smartphone-Collected Speech Corpora," in In Proc. 3rd International 
Workshop on Spoken Language Technologies for Under-resourced Languages 
(SLTU), Cape Town, South Africa, May 2012, pp. 68–75.

[6]        C van Heerden, M.H. Davel and E. Barnard, "Medium-Vocabulary Speech 
Recognition for Under-Resourced Languages", in In Proc. 3rd 
International Workshop on Spoken Language Technologies for 
Under-resourced Languages (SLTU), Cape Town, South Africa, May 2012, pp. 
146-151.

[7]        J. Badenhorst, A. De Waal  and F. de Wet, "Quality measurements for  
mobile data collection in the developing world", in In Proc. 3rd 
International Workshop on Spoken Language Technologies for 
Under-resourced Languages (SLTU), Cape Town, South Africa, May 2012, pp. 
139-145.

--------------------------------------------------------------------------------
CSIR Human Language Technology
--------------------------------------------------------------------------------
