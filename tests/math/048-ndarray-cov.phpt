--TEST--
NDArray::cov
--FILE--
<?php
$a = \NDArray::array([[3, 7, 8], [2, 4, 3]]);
print_r(\NDArray::cov($a)->toArray());
$b = \NDArray::array([[1, 2, 3, 4], [5, 4, 3, 2]]);
print_r(\NDArray::cov($b)->toArray());
$c = \NDArray::array([[1, 2, 3, 4], [5, 6, 7, 8]]);
print_r(\NDArray::cov($c)->toArray());
$d = \NDArray::array([[1, 2, 3, 4], [1, 2, 3, 4]]);
print_r(\NDArray::cov($d)->toArray());
$e = \NDArray::array([[1, 2, 3, 4]]);
print_r(\NDArray::cov($e)->toArray());
$f = \NDArray::array([[0, 0, 0, 0], [0, 0, 0, 0]]);
print_r(\NDArray::cov($f)->toArray());
$g = \NDArray::array([[3, 7, 8], [2, 4, 3]]);
print_r(\NDArray::cov($g, False)->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 7
            [1] => 2
        )

    [1] => Array
        (
            [0] => 2
            [1] => 1
        )

)
Array
(
    [0] => Array
        (
            [0] => 1.6666666269302
            [1] => -1.6666666269302
        )

    [1] => Array
        (
            [0] => -1.6666666269302
            [1] => 1.6666666269302
        )

)
Array
(
    [0] => Array
        (
            [0] => 1.6666666269302
            [1] => 1.6666666269302
        )

    [1] => Array
        (
            [0] => 1.6666666269302
            [1] => 1.6666666269302
        )

)
Array
(
    [0] => Array
        (
            [0] => 1.6666666269302
            [1] => 1.6666666269302
        )

    [1] => Array
        (
            [0] => 1.6666666269302
            [1] => 1.6666666269302
        )

)
Array
(
    [0] => Array
        (
            [0] => 1.6666666269302
        )

)
Array
(
    [0] => Array
        (
            [0] => 0
            [1] => 0
        )

    [1] => Array
        (
            [0] => 0
            [1] => 0
        )

)
Array
(
    [0] => Array
        (
            [0] => 0.5
            [1] => 1.5
            [2] => 2.5
        )

    [1] => Array
        (
            [0] => 1.5
            [1] => 4.5
            [2] => 7.5
        )

    [2] => Array
        (
            [0] => 2.5
            [1] => 7.5
            [2] => 12.5
        )

)