--TEST--
NDArray::divide
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
print_r(($a / 2)->toArray());
print_r(($a / $a)->toArray());
print_r(($a / $a[0])->toArray());
print_r(($a / [[1],[2]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 0.5
            [1] => 1
        )

    [1] => Array
        (
            [0] => 1.5
            [1] => 2
        )

)
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 1
        )

    [1] => Array
        (
            [0] => 1
            [1] => 1
        )

)
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 1
        )

    [1] => Array
        (
            [0] => 3
            [1] => 2
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
            [0] => 1.5
            [1] => 2
        )

)