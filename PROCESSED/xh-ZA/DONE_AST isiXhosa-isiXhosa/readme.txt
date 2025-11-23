AST Disk 10

This disk contains African Speech Technology speech and transcription data for the XX database.  The "speech" directory contains Xhosa speech as spoken by Xhosa mother tongue speakers.

The disk has the following directory structure:

XX
|-- lexicons
|   |-- astbet
|   `-- xsampa
|-- speech
|   `-- alaw
`-- transcriptions
    |-- ast
    |   |-- astbet
    |   `-- xsampa
    |-- htk
    |   `-- astbet
    `-- praat
        `-- xsampa

The "lexicons" directory has two subdirectories -- "astbet" and "xsampa".  The "xsampa" directory contains the following ten files:
XX_all_count.txt
XX_all_lexicon.txt
XX_all_sourcefiles.txt
XX_exotic_count.txt
XX_exotic_lexicon.txt
XX_exotic_sourcefiles.txt
XX_native_count.txt
XX_native_lexicon.txt
XX_native_sourcefiles.txt
XX_transcriptions.txt

The astbet directory contains the following three files:
XX_all_lexicon.txt
XX_exotic_lexicon.txt
XX_native_lexicon.txt

All speech and non-speech was transcribed orthographically and phonetically according to the AST Transcription Specifications.
Speech in any language other than the target languages was encapsulated in ($) (/$) symbols.  Three sets of lexicons are provided in the "Lexicons" directory.  *_all_* refer to all speech including the target language and any other languages.  *_native_* refer to speech in the target language.
*_exotic_* refer to speech in any language other than the target language.

The *_lexicon.txt file contains a variation pronunciation lexicon for all words and word fragments as they occur in a particular database.  The *_counts.txt file holds the phonetic sequence counts, i.e. the number of times each particular pronunciation occurs in the TextGrid files.  The data is presented in the same structure used to present the pronunciation lexicon.  Therefore, the counts will be written to the locations where the phonetic sequences would occur in the lexicon.
The *_sourcefiles.txt file holds the information needed to locate the TextGrid files that are associated with each phonetic sequence in the pronunciation lexicon.  

The "speech" directory has one subdirectory ("alaw") which contains the speech data in A-law format.

The transcription files are in three different formats -- the "ast" directory contains files in an AST specific format (TRC02), the "htk" directory contains files in the custom format of HTK and the praat directory contains TextGrid files for use with Praat.

The TRC02 files consist of five columns which have the following names and meanings:

Name			Meaning

ETimes			Segment end time
Boundary_Type		"manually_placed" or "equally_divided".
Phone			The phone symbol (in XSampa or AstBet)
Word			The word
Word_Type		A classification for the word (see below)

"equally_divided" boundaries are produced by equally dividing the space between two manual boundaries by the number of phones between the two manual boundaries.  Therefore, "equally_divided" boundaries are not at their "correct" location and should be regarded as placeholders.  The "manually_placed" boundaries were placed by human transcribers.

The "Word_Type" column in the TRC02 files uses the following naming convention:

Codeword	       Meaning

filler		       Filler word (e.g. uhm)
psl		       Phonetically spelled word
trunc		       Truncated word
frag		       Word fragment
mispr		       Mispronounced word
assim		       Assimilation
splabb		       Spelled abbreviation
spllet		       Spelled letter (e.g. an initial)
splwrd		       Spelled word
acron		       Acronym (said as a word)
name		       Name of a person, street, town, etc.
common		       Common word

All text files have DOS line terminators (\r\n), so you should run dos2unix on these files for use on Unix systems.
