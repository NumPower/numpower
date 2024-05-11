--TEST--
NDArray::zeros
--FILE--
<?php
$a = \NDArray::zeros([4]);
print_r($a->toArray());
$a = \NDArray::zeros([4, 4]);
print_r($a->toArray());
?>
--EXPECT--
Array
(
    [0] => 0
    [1] => 0
    [2] => 0
    [3] => 0
)
Array
(
    [0] => Array
        (
            [0] => 0
            [1] => 0
            [2] => 0
            [3] => 0
        )

    [1] => Array
        (
            [0] => 0
            [1] => 0
            [2] => 0
            [3] => 0
        )

    [2] => Array
        (
            [0] => 0
            [1] => 0
            [2] => 0
            [3] => 0
        )

    [3] => Array
        (
            [0] => 0
            [1] => 0
            [2] => 0
            [3] => 0
        )

)