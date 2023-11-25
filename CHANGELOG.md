# CHANGELOG



## v0.1.3 (2023-11-25)

### Build

* build: Update .readthedocs.yml ([`5b2311a`](https://github.com/ellagale/doenut/commit/5b2311a06c3ba101beb4e4e8c8e9351efc147183))

* build: add the rtd configuration ([`cc34ffd`](https://github.com/ellagale/doenut/commit/cc34ffd97c6d73114cd978afc00cb9a26b6df1b2))


## v0.1.2 (2023-11-25)

### Build

* build: switch back to napoleon ([`fda2e78`](https://github.com/ellagale/doenut/commit/fda2e78cea3657b8ff62e2595340ceebdcf96cf9))

* build: add epytext module for doc generation ([`4f7a8d5`](https://github.com/ellagale/doenut/commit/4f7a8d5183f9230703633de9591577ac845a47e3))

* build: update poetry for documents ([`1e9b2db`](https://github.com/ellagale/doenut/commit/1e9b2db48a5ab26a6b5dfd074cf3eabec3525355))

### Chore

* chore: fix line endings on changelog ([`f4b5643`](https://github.com/ellagale/doenut/commit/f4b5643d0c588ac5bdc9e7babbf18fa0894d12fd))

### Fix

* fix: handle series data in datasets ([`353f50c`](https://github.com/ellagale/doenut/commit/353f50c52a8fce13d66da33e419c54658a449fad))

* fix: Not all functions of doenut.py were being exported. ([`304047f`](https://github.com/ellagale/doenut/commit/304047f885c4417bec6b67c4bf73d8d3a407b457))

### Unknown

* complete basic docs ([`e24bc40`](https://github.com/ellagale/doenut/commit/e24bc402f612792bba06374d41eb5cba1b61b040))

* doc: yet more docstrings ([`d0f1a4c`](https://github.com/ellagale/doenut/commit/d0f1a4c21c21fa9a52915c3232bc990331bc8ccc))

* doc: yet more docstrings ([`8d29f05`](https://github.com/ellagale/doenut/commit/8d29f0594042a1884d9800b7f1970ba5a6823e9b))

* doc: more docstrings ([`6c53e8c`](https://github.com/ellagale/doenut/commit/6c53e8cf7b5e6146ba55fde52e7e0d381c70eef3))

* doc: formatting ([`9d01fbb`](https://github.com/ellagale/doenut/commit/9d01fbbb270ddc2523509ba5b858f8cc771a2ace))

* doc: many more docstrings ([`f902b94`](https://github.com/ellagale/doenut/commit/f902b9439dd1a7ec3d325b58d87476f446ed7e60))

* doc: wip on tidying docstrings for publishing ([`dafff2f`](https://github.com/ellagale/doenut/commit/dafff2f794a296d8007bfbbde25b239ef94de32d))

* doc: add basic generated docs ([`ffdfbc4`](https://github.com/ellagale/doenut/commit/ffdfbc48ab3f2462309b1fd0a81f8230a57674ac))

* add default code of conduct / contrib files (#31) ([`5242f44`](https://github.com/ellagale/doenut/commit/5242f4438d465d37d26e9fa866bd53611b9c2876))

* 26 proper logging (#30)

* feat: added basic logging framework

* feat: more logging implemented

* feat: finish adding basic logging across package. ([`2eb036d`](https://github.com/ellagale/doenut/commit/2eb036d553b7f68c807a4e536f6057919059981e))


## v0.1.1 (2023-11-18)

### Build

* build: bump to v0.1.1 ([`610efa0`](https://github.com/ellagale/doenut/commit/610efa0b6b1ec9f8801ce979135b840a0c3d09e0))

* build: prep for v0.1.1 ([`0b4b517`](https://github.com/ellagale/doenut/commit/0b4b517a68d557e813cb758a69267464ba4a6ef3))

### Unknown

* 22 check all the docstrings help docs (#28)

* fix: column_selector should allow response only filtering

* fix: scaling helper function has wrong default for optional

* doc: A lot of docstrings

* doc: better readme

* chore: add semantic release ([`74a32ff`](https://github.com/ellagale/doenut/commit/74a32ff1226d93b16492d6fed8dad0e451b5d96a))


## v0.1.0 (2023-11-17)

### Build

* build: prepare for v0.1.0 release ([`77049c1`](https://github.com/ellagale/doenut/commit/77049c1f0cc7c1b0821bda520a25c00d32a99d0c))

### Unknown

* fix designer ([`267b3ff`](https://github.com/ellagale/doenut/commit/267b3ff3b04f33aaf88384f66525cc28476856b9))

* add tune_model to averaged_model. Oh and black now supports ipynb ([`b825e26`](https://github.com/ellagale/doenut/commit/b825e26811a3b62fb8b4c4709ac873682bc059c7))

* Set scaling to not break POLS by only scaling inputs by default ([`46b3ef2`](https://github.com/ellagale/doenut/commit/46b3ef2c313e3a5ea7f7cc8678366e93b3158f76))

* 18 refactor into classes (#19)

* move plotting into its own file

* Doenut is no longer calulated.

* add Model and AveragedModel. Make the function formerly known as calulate use the new model code

* make scaling per-column

* add documentation to model code

* restore distinction between where scaling can be done.

* Add first pass at calulating a selective model + a test for it.

* Don&#39;t need to pass the response key in, we know it.

* tidy up some imports

* hive model set functionality from averaged model off into its own class

* rework of model sets

* got input_selector working on averaged model

* further work on classification. Sort out sequencing of R2/Q2 calculations

* added autotune test

* autotune now calls averagemodel directly

* add filterable dataframe helper

* rename and move terms into new package

* Tweaked data so averaged duplicate is actually different to original data.
implemented duplicate removal and averaging in filtered data frame

* add average / remove duplicates to DataFrameSet

* make more sense of the dataset object. duplicates handled OK, need to finish averaging code

* add builder pattern to set functions in DataSet, tidy up typing.

* had an annoying idea to make this better.

* start of insane rewrite of classes

* Basic dataset and filtered dataset implemented and tests added

* rework (again) to match builder pattern

* add scaling modifier

* rename / sort the modifiers into their own module, add duplicate handling modifiers

* Add tests for duplicate handling

* work on updating model classes to use new dataset

* further work on fixing models

* two tests left

* one test remaining

* added a bunch of documentation to the code.

* make all the modifiers return a new object.

* split data_set into data_set and modifiable_data_set

* allow for deciding if responses should be scaled

* autotune now works

* clean up defunct code from doenut.py (mainly)

* test on python 3.10

* hopefully fix test on python 3.10

* further typing fixes for earlier pythons

* missed some

* further tidying up. Updated the Test 2 example

* update manual ([`53f21c9`](https://github.com/ellagale/doenut/commit/53f21c95edfe47c3978a11455b06a46e1a345e54))

* 4 remove doepy (#12)

* implement new version of full_fact

* revert full fact to taking a dictionary

* add test for fractional model

* add wrapper around fract fact ([`b258ffa`](https://github.com/ellagale/doenut/commit/b258ffaa6786727687a7778dbe300a45895e9b9f))

* enable pytest in github actions (#11) ([`ec95d49`](https://github.com/ellagale/doenut/commit/ec95d49709916bc5fb49c42a151249ec7639ee15))

* Merge pull request #9 from ellagale/6-create-basic-workflow-demonstration-regression-test

6 create basic workflow demonstration regression test ([`b83f58b`](https://github.com/ellagale/doenut/commit/b83f58bbf1fb6f7f034aa2aa031392267c716abb))

* work on docstrings, reformating ([`d17d374`](https://github.com/ellagale/doenut/commit/d17d3745f3d8c702c11c09ca0c07d2f1b945b52d))

* reformat + lint tests ([`3a7f59f`](https://github.com/ellagale/doenut/commit/3a7f59fac873f33f0739c1c18d104df00b4cc0e8))

* add last of model validation ([`4e95ba0`](https://github.com/ellagale/doenut/commit/4e95ba0e8aa19ae212c8e39d931ffc8733801893))

* complete second set of tests ([`a868b7d`](https://github.com/ellagale/doenut/commit/a868b7d25d35616369b2bbb93509ffa7d2a11a03))

* further work on doenut test. Numbers need validation. ([`479a043`](https://github.com/ellagale/doenut/commit/479a04355bc4d54c59ab61ffe3a7309bd8f86be7))

* work on removing warnings ([`1ca0b4c`](https://github.com/ellagale/doenut/commit/1ca0b4cbf9c5c8052e247c01514acaaa2df8a6bd))

* remove missing parameter errors ([`ebcd86c`](https://github.com/ellagale/doenut/commit/ebcd86cbfa467e1f8944852a08d4362ba9dc19b1))

* test basic parsimonious model ([`ef20d0e`](https://github.com/ellagale/doenut/commit/ef20d0e2f3b578ea518f0660ed006322f7f7b0f9))

* test basic full model ([`763c1c0`](https://github.com/ellagale/doenut/commit/763c1c0d3da1ec937f3b9dc8d800ec0ac09bc21c))

* test basic R2/Q2 calculation ([`a0b559b`](https://github.com/ellagale/doenut/commit/a0b559bc7e831dcd070fe9b4c2bf49a22e9acb5d))

* update a couple of functions for new pandas changes ([`fac2b0c`](https://github.com/ellagale/doenut/commit/fac2b0c23ff4cf691ffa6c5352ae6856870e7dd5))

* add initial test ([`16101ae`](https://github.com/ellagale/doenut/commit/16101aed820ab95d61ea017b0b6841a824b94f6d))

* switch to poetry from setuptools ([`29ca60a`](https://github.com/ellagale/doenut/commit/29ca60ad791631730c28999f125bad2a02eed72c))

* remove pycharm files from repo ([`d032cea`](https://github.com/ellagale/doenut/commit/d032cea743839775bb00a8c09faefe86494c0776))

* Merge pull request #8 from ellagale/3-switch-license-to-mit

switch license to MIT ([`c50314c`](https://github.com/ellagale/doenut/commit/c50314c5c4c64f9c5e3fae5d32df488adbcc0b63))

* switch license to MIT ([`b5fedc8`](https://github.com/ellagale/doenut/commit/b5fedc802fc2b15d246b7f712a3fd02968c9efe9))

* add rework warning to readme ([`f52a3ad`](https://github.com/ellagale/doenut/commit/f52a3adeae6e4fcc4ec5a83813edecb0f0791ca8))

* Merge pull request #2 from cwoac/main

Turn doenut into a module pt1 ([`71680da`](https://github.com/ellagale/doenut/commit/71680da531244c51f4f3bf2396a1d8e4b99f80eb))

* pep8 on designer ([`4e92cf2`](https://github.com/ellagale/doenut/commit/4e92cf2aad516d86153dc1737032da499451549f))

* add designer module ([`fa7af02`](https://github.com/ellagale/doenut/commit/fa7af02a50fc8add242348eeab0f589bde12ed0d))

* some tidying up ([`591a825`](https://github.com/ellagale/doenut/commit/591a82507c55fb57ba2d9cf0af35a4e3f810d742))

* ignore checkpoints ([`e231eec`](https://github.com/ellagale/doenut/commit/e231eec79e5e4c8a252060045a48460d4b3b6f98))

* get the backronym right ([`2f77fc1`](https://github.com/ellagale/doenut/commit/2f77fc1a03c573540a6dc8170919ec824cbc57af))

* start work on tidying up doenut.py ([`b8138da`](https://github.com/ellagale/doenut/commit/b8138da72049a563bf66c069f72c6299dec4c6af))

* pep8 setup ([`b49bec6`](https://github.com/ellagale/doenut/commit/b49bec696f8d40caaebffb66a048c99f9a8d052a))

* ignore dist directory ([`0d92043`](https://github.com/ellagale/doenut/commit/0d9204396031903b0f2092d6d89780e9d6c55769))

* basic module setup ([`262e0e0`](https://github.com/ellagale/doenut/commit/262e0e0bc3ea250fe4e76b054870923694de9ccc))

* ignore build files ([`01d2daa`](https://github.com/ellagale/doenut/commit/01d2daa4218d72165389abd96cf79feb6188e25d))

* don&#39;t commit cache files ([`4706ee4`](https://github.com/ellagale/doenut/commit/4706ee4e55405e5111ef999cc50d2dcafb4af144))

* move doenut to module dir ([`e7e3bf2`](https://github.com/ellagale/doenut/commit/e7e3bf26bd74fead6ba55a3c1f4d55381ebdb888))

* Added in the original solar cell tutorial ([`e16e661`](https://github.com/ellagale/doenut/commit/e16e6615e223beaf277882d5e31730469d5c12bd))

* updates? ([`48aa179`](https://github.com/ellagale/doenut/commit/48aa17991daa414820f01990ca9705fe46af662b))

* Update doenut.py ([`919025a`](https://github.com/ellagale/doenut/commit/919025aaa8314ce392f7784db3343d71d69aaef8))

* Update doenut.py ([`11bed10`](https://github.com/ellagale/doenut/commit/11bed10639934fc766100c9a37147277cf363b0e))

* corrected some errors ([`dc012d8`](https://github.com/ellagale/doenut/commit/dc012d874a48d6ee1c8dd22a4010ca56b3a3ab1e))

* deepcopy fixes a slot of errors! ([`f7dc0e4`](https://github.com/ellagale/doenut/commit/f7dc0e47b5506bb72dbc6cb8758304901749e129))

* fixed an error with normalised coefficients ([`1d8768a`](https://github.com/ellagale/doenut/commit/1d8768a48ae693405e48c750ee87290413f07d63))

* Made experimental designer submodules ([`ca54105`](https://github.com/ellagale/doenut/commit/ca54105457fa6ba9c1bd9405b682482526f9c6cf))

* Update Forthcoming_features.md ([`21d4d4e`](https://github.com/ellagale/doenut/commit/21d4d4edffb6b3c0952f9f8db45dbebfb8adc29b))

* Update Manual.md ([`ee4403a`](https://github.com/ellagale/doenut/commit/ee4403abd80e08c7e37a87755d397952bfbb5662))

* Update README.md ([`f84e908`](https://github.com/ellagale/doenut/commit/f84e9082bdbe82a0247dcab6b0a07906040834f6))

* Update README.md ([`f0d7b74`](https://github.com/ellagale/doenut/commit/f0d7b74cb9c6d0d3c0c4d740c140a2256ef0d0bf))

* Update README.md ([`a3f0de2`](https://github.com/ellagale/doenut/commit/a3f0de2f0ab6f0020012275b1dee3a86c874d2cd))

* Update README.md ([`da79cf3`](https://github.com/ellagale/doenut/commit/da79cf3b552ba6380e8b3be99a324e4843d82217))

* Update README.md ([`743bf75`](https://github.com/ellagale/doenut/commit/743bf75c34d5d188600878e0df9dc67321076915))

* Update README.md ([`d30a1f4`](https://github.com/ellagale/doenut/commit/d30a1f428498023df7f07e5473b9c1149d32ce44))

* Update README.md ([`ea8b57d`](https://github.com/ellagale/doenut/commit/ea8b57d5ef64dc6ed40dfa6164d0b84b72fe7732))

* Update Manual.md ([`d9864ac`](https://github.com/ellagale/doenut/commit/d9864ac23cf356c2e74037533a38e0371e8f3985))

* Update Forthcoming_features.md ([`ae576ac`](https://github.com/ellagale/doenut/commit/ae576ac201f4aa7307fd2fb8fa2f03258b17a7b8))

* Update Manual.md ([`423e78a`](https://github.com/ellagale/doenut/commit/423e78a34d64e6d98483400053ababde702951f1))

* Update Manual.md ([`a321db7`](https://github.com/ellagale/doenut/commit/a321db7430146c180fa836f401d7c03a45140cf3))

* Merge branch &#39;main&#39; of https://github.com/ellagale/doenut ([`390259c`](https://github.com/ellagale/doenut/commit/390259cacaf8a6a396ab5e58522786bd01ba419c))

* Create doenut_small.jpg ([`43c2002`](https://github.com/ellagale/doenut/commit/43c20022769e4bf4d5674c50db1f9c3de7942005))

* Update Manual.md ([`2bb598f`](https://github.com/ellagale/doenut/commit/2bb598fe0b6a952516f1124fcf4fb4cb55489d3f))

* Update Manual.md ([`ac68c54`](https://github.com/ellagale/doenut/commit/ac68c54288042e8e7d9238d61ab134f2c18b3802))

* Update Manual.md ([`3591490`](https://github.com/ellagale/doenut/commit/359149000a9018004be9bdcbe01fa02348665130))

* Merge branch &#39;main&#39; of https://github.com/ellagale/doenut ([`573c65b`](https://github.com/ellagale/doenut/commit/573c65b54c15a04391922add20938607d3dbea1f))

* made stupid logo ([`7fd7018`](https://github.com/ellagale/doenut/commit/7fd7018ee7c273c84c02aa5bbde715df3d648bbc))

* Update Forthcoming_features.md ([`2a9357a`](https://github.com/ellagale/doenut/commit/2a9357a9e222d7de22823bbb068cf8cdf82ed5b7))

* Update Forthcoming_features.md ([`4bbc23a`](https://github.com/ellagale/doenut/commit/4bbc23a756bc879da32f5974ff2d0e88f038afb7))

* Create Forthcoming_features.md ([`dd5ad7e`](https://github.com/ellagale/doenut/commit/dd5ad7eb0295280b9692b602c75ce68bad7dca59))

* Merge pull request #1 from ellagale/add-license-1

Create LICENSE ([`f174e8a`](https://github.com/ellagale/doenut/commit/f174e8a917ba0bed5d9546f2317bc2b333d1257b))

* Create LICENSE ([`7471497`](https://github.com/ellagale/doenut/commit/74714976cffb1f273999faa507aa67329c2dfa30))

* Update Manual.md ([`8b3becd`](https://github.com/ellagale/doenut/commit/8b3becdec6a45201aff517213dc9a41e5056c16f))

* Update Manual.md ([`e1c3a1d`](https://github.com/ellagale/doenut/commit/e1c3a1dcd2f6c059fd95d7fc0ec57c71840f1ac5))

* moved images ([`25f66d7`](https://github.com/ellagale/doenut/commit/25f66d784618c607d2f91c656c2e656a0b19f824))

* Update Manual.md ([`1d54b24`](https://github.com/ellagale/doenut/commit/1d54b2441789c323ab560eb9f89ebd1fa068e775))

* Found missing grapjhs ([`03312fa`](https://github.com/ellagale/doenut/commit/03312fa32167ca48b8f1c60d9d0a845cd66a94ac))

* Update Manual.md ([`89e76a5`](https://github.com/ellagale/doenut/commit/89e76a5d01619699b9082a02c569f49b47a8d3f4))

* Update Manual.md ([`f7bba17`](https://github.com/ellagale/doenut/commit/f7bba17048946c414eaecf11ce7780e0f6ed93d5))

* Update Manual.md ([`811277b`](https://github.com/ellagale/doenut/commit/811277bbc9c2637b159fa1ca58f1082d624958d5))

* Update Manual.md ([`5bfd0f9`](https://github.com/ellagale/doenut/commit/5bfd0f9df2469dd347e59ef3e7188d42ff4618ef))

* Added graphs for manual ([`eed37c8`](https://github.com/ellagale/doenut/commit/eed37c8766ea49711ac550610dfd314782253b5c))

* Update Manual.md ([`e1419e6`](https://github.com/ellagale/doenut/commit/e1419e6219c8c65b078ba6c543260aa708b334e6))

* Create Manual.md ([`c59ee22`](https://github.com/ellagale/doenut/commit/c59ee2213879c17206cf22b46c5d269cd6058c4f))

* Update README.md ([`178861e`](https://github.com/ellagale/doenut/commit/178861e59ebe431a2afbfac73a27841daa6b646c))

* Update README.md ([`dbc1334`](https://github.com/ellagale/doenut/commit/dbc133479decb0e2d47a59216589d6b067d9a735))

* Removed 3rd year lab answers ([`2e48bdc`](https://github.com/ellagale/doenut/commit/2e48bdce7899011fae8a1dd74a39f776e9466066))

* Merge branch &#39;main&#39; of https://github.com/ellagale/doenut ([`360efb5`](https://github.com/ellagale/doenut/commit/360efb54c23f3df4cceb86476d1bb14af886504d))

* Added the changes made working for Pfizer ([`2bed225`](https://github.com/ellagale/doenut/commit/2bed225b2abeef8c6a4e05dcd7bfce2648351636))

* Create README.md ([`694ea9e`](https://github.com/ellagale/doenut/commit/694ea9e46b6b841ad280958078555bed4b518a4c))

* added in error plots

Added in error plots ([`15fd46d`](https://github.com/ellagale/doenut/commit/15fd46d9c928fbc4078602e4767a70318ec5f79f))

* update ([`127c80d`](https://github.com/ellagale/doenut/commit/127c80d4ec8cff19260afe1f1d6a939d0bfaa2da))

* update ([`5bc39c5`](https://github.com/ellagale/doenut/commit/5bc39c5808688e39ca58b6f55a73b9dadcd4b48d))

* update ([`7e6967f`](https://github.com/ellagale/doenut/commit/7e6967fe513953b8a4daa90deb8704d555f291da))

* initial commit ([`3227667`](https://github.com/ellagale/doenut/commit/3227667d9055f10b81f769737a154057f662010d))
