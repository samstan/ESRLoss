# ESRLoss
Code for ESR loss function in offline contextual bandits

For the Yahoo dataset, download the dataset from https://webscope.sandbox.yahoo.com/catalog.php?datatype=r&did=49.

Then run parse.py on the extracted files (of which there are ten). Then run ads.py, which relies on learners_news.py to train models.

For the IDHP dataset, use generate_data.py to first generate the "A" setting dataset. Then run ihdp_reg.py, which relies on learners_reg.py to train models.
