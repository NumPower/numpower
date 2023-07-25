---
name: Bug report
about: Create a report to help us improve
title: "[BUG]"
labels: bug
assignees: henrique-borba

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior

**Expected behavior**
A clear and concise description of what you expected to happen.

**Dumps**
If applicable, add NDArray low-level dumps of the relevant arrays. 

```php
$a->dump();
```
```
=================================================
NDArray.uuid                    0
NDArray.dims                    [ 1 4 ]
NDArray.strides                 [ 16 4 ]
NDArray.ndim                    2
NDArray.device                  CPU
NDArray.refcount                1
NDArray.descriptor.elsize       4
NDArray.descriptor.numElements  4
NDArray.descriptor.type         float32
=================================================
```

**Environment (please complete the following information):**
 - OS: [e.g. Ubuntu 20.04]
 - PHP Version: [e.g. 8.2]
 - NumPower Version: [e.g. master, 0.1.2]
 - Compiled with CUDA (--with-cuda): [e.g. Yes, No, Idk]
 - Compiled with AVX2: [e.g. Yes, No, Idk]

**Optional: PHP Information**
PHP Version Info (php -v)
```
PHP 8.2.0 (cli) (built: Jul 25 2023 15:25:01) (NTS DEBUG)
Copyright (c) The PHP Group
Zend Engine v4.2.0, Copyright (c) Zend Technologies
```
PHP Modules:
```
[PHP Modules]
bz2
Core
ctype
curl
date
dom
fileinfo
filter
hash
iconv
json
libxml
mbstring
mysqlnd
NumPower
openssl
pcre
PDO
pdo_mysql
pdo_sqlite
Phar
posix
random
Reflection
session
SimpleXML
SPL
sqlite3
standard
tokenizer
xml
xmlreader
xmlwriter
zip
zlib

[Zend Modules]
```

**Additional context**
Add any other context about the problem here.
