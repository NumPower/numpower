--TEST--
NDArray::array
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
print_r($a->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 2
        )

    [1] => Array
        (
            [0] => 3
            [1] => 4
        )

)
