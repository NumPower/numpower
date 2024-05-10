--TEST--
NDArray::multiply
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
print_r(($a * 2)->toArray());
print_r(($a * $a)->toArray());
print_r(($a * $a[0])->toArray());
print_r(($a * [[1],[2]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 2
            [1] => 4
        )

    [1] => Array
        (
            [0] => 6
            [1] => 8
        )

)
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 4
        )

    [1] => Array
        (
            [0] => 9
            [1] => 16
        )

)
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 4
        )

    [1] => Array
        (
            [0] => 3
            [1] => 8
        )

)
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 2
        )

    [1] => Array
        (
            [0] => 6
            [1] => 8
        )

)