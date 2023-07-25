--TEST--
NDArray::sinh
--FILE--
<?php
$a = \NDArray::array([[0, -0.5], [0, -0.5]]);
print_r(\NDArray::sinh($a)->toArray());
print_r(\NDArray::sinh($a[0])->toArray());
print_r(\NDArray::sinh([[0],[-0.5]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 0
            [1] => -0.52109527587891
        )

    [1] => Array
        (
            [0] => 0
            [1] => -0.52109527587891
        )

)
Array
(
    [0] => 0
    [1] => -0.52109527587891
)
Array
(
    [0] => Array
        (
            [0] => 0
        )

    [1] => Array
        (
            [0] => -0.52109527587891
        )

)