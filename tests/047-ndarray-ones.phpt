--TEST--
NDArray::ones
--FILE--
<?php
$a = \NDArray::ones([4]);
print_r($a->toArray());
$a = \NDArray::ones([4, 4]);
print_r($a->toArray());
?>
--EXPECT--
Array
(
    [0] => 1
    [1] => 1
    [2] => 1
    [3] => 1
)
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 1
            [2] => 1
            [3] => 1
        )

    [1] => Array
        (
            [0] => 1
            [1] => 1
            [2] => 1
            [3] => 1
        )

    [2] => Array
        (
            [0] => 1
            [1] => 1
            [2] => 1
            [3] => 1
        )

    [3] => Array
        (
            [0] => 1
            [1] => 1
            [2] => 1
            [3] => 1
        )

)