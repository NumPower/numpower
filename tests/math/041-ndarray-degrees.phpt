--TEST--
NDArray::degrees
--FILE--
<?php
$a = \NDArray::array([[0, -0.5], [0, -0.5]]);
print_r(\NDArray::degrees($a)->toArray());
print_r(\NDArray::degrees($a[0])->toArray());
print_r(\NDArray::degrees([[0],[-0.5]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 0
            [1] => -28.647890090942
        )

    [1] => Array
        (
            [0] => 0
            [1] => -28.647890090942
        )

)
Array
(
    [0] => 0
    [1] => -28.647890090942
)
Array
(
    [0] => Array
        (
            [0] => 0
        )

    [1] => Array
        (
            [0] => -28.647890090942
        )

)