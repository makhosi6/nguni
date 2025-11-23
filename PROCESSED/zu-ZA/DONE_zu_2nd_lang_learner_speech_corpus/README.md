
# isiZulu Second Language Learner Speech Corpus

This corpus is specifically designed to assist in evaluating the performance of pronunciation feedback tools for second language learning. The corpus is comprised of gold standard recordings from isiZulu teachers (2,493 recordings) and recordings from isiZulu L2 learners that have been annotated by isiZulu teachers for phonemic and tonal pronunciation errors (9,639 recordings). The accompanying database and tsv file include the teacher annotations and demographic information. 

## Sentence Source
The sentences in this corpus primarily come from the textbook, *Elementary Zulu: A Course of Elementary Lessons in the Zulu Language: Intended Chiefly for Beginners and Junior* by M.F.W. 1921. OCR was run on the book to automatically extract example sentences. Regular expressions were used to standardize to the modern orthography. This list was reviewed in consult with a Zulu language teacher to update any archaic usages of language. Additionally, 20 sentences that are common in language learning environments and 8 sentences that included phonemes that were underrepresented in the corpus were added. 

## Filename structure
The filenames represent the elicitaion order, the unit of origin from the textbook, and the speaker id. For example, a recording with the filename *819-22-102.wav* would be the 819th sentence recorded, coming from the 22nd chapter of the textbook, spoken by paricipant 102. The 20 classroom phrases are marked with PHREX and the 8 underrepresented phoneme phrases were marked with PHON instead of chapter numbers.

## Annotation error format
Each sentence elicited from a student has 1-3 annotations for errors. The phonemic errors are marked binarily, with a 1 indicating correct pronunciation and a zero representing incorrect pronunciation. Phoneme(s) insertion is marked as an index, with the index indicating the position of insertion in the sequence of phonemes in a sentence. 0 would indicate an insertion before the first phoneme, such as a student pronouncing "ng.i.m.b.o.ng.i.l.e." with a vowel at the beginning, such that it sounds like "e.ng.i.m.b.o.ng.i.l.e." An insertion marked as 1 would be the addition of a sound before the second phoneme of the sentence, and so on. Tonal errors are indexed on the sequence of syllables in the sentence. For example, tonal error marked on "ngi.ya.ku.kho.lwa." with annotation of ["3","4"], would indicate that the student produced the incorrect tone on "ku" and "kho".

## Corpus recording information
The recordings in this corpus were collected from June-July 2023 at the University of KwaZulu-Natal Edgewood Campus in Durban, South Africa. Participants were compensated for the recordings and breaks were encouraged between sets of 50 sentences. The recordings were done in empty classrooms on researchers' laptops.

##Accompanying Files
The database file is an SQLite DB file and contains 3 tables: clips, feedback and speakers. The clips table contains each the name of each file in the corpus, the text of the sentence, and the speaker that read the sentence. The feedback table contains the filenames, the id of the teacher that graded the audio clip, a string of binary phoneme correctness judgements that align sequentially with the phonemes in the sentence, a tone score that notes the syllable index of a tone error, and a sound(s) segment insertion error that represents the insertion of one or more sounds immediately preceeding (to the left of) the referenced phoneme of the index. The speakers table contains demographic information about the corpus partipants, including their speaker ID, gender, first language (L1), other languages spoken, semesters of study, place of residency, birthplace, pre-university years of studying isiZulu, and their age. The accompanying metadata tsv file is a simpler extraction of all of this data for users that are uncomfortable with SQLite DB files.

##Further information
For further information about this corpus, [see the LREC-COLING 2024 conference proceedings publication](https://aclanthology.org/2024.lrec-main.429/) titled "Developing a Benchmark for Pronunciation Feedback: Creation of a Phonemically Annotated Speech Corpus of isiZulu Language Learner Speech" by Alexandra O'Neil, Nils Hjortnaes, Zinhle Nkosi, Thulile Ndlovu, Zanele Mlondo, Ngami Phumzile Pewa, and Francis Tyers or send an email to aconeil@iu.edu.

