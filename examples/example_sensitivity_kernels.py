#!/usr/bin/env python3
"""
Example: Using Real Sensitivity Kernels

This script demonstrates how to use the SensitivityKernelProvider
to work with real seismological kernel data.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from pygeoinf.interval import (
    SensitivityKernelCatalog,
    SensitivityKernelProvider,
    IntervalDomain,
    Lebesgue,
    DepthCoordinateSystem
)


def main():
    """Run example demonstrations."""

    # Path to kernel data
    data_dir = Path("pygeoinf/interval/demos/kernels_modeplotaat_Adrian")

    if not data_dir.exists():
        print(f"Error: Kernel data directory not found: {data_dir}")
        print("Please ensure the kernel data is available.")
        return

    print("=" * 70)
    print("Real Sensitivity Kernel Example")
    print("=" * 70)

    # =========================================================================
    # Step 1: Initialize catalog
    # =========================================================================
    print("\n1. Initializing kernel catalog...")
    catalog = SensitivityKernelCatalog(data_dir)

    print(f"   Found {len(catalog)} modes")
    print(f"   Overtone range: n = {catalog.get_catalog_summary()['n_range']}")
    print(f"   Angular order range: l = {catalog.get_catalog_summary()['l_range']}")

    # Get period range
    try:
        T_min, T_max = catalog.get_period_range()
        print(f"   Period range: {T_min:.1f} - {T_max:.1f} s")
    except ValueError:
        print("   Period information not available")

    # =========================================================================
    # Step 2: Explore specific mode
    # =========================================================================
    print("\n2. Exploring mode 00s03...")
    mode = catalog.get_mode("00s03")

    print(f"   Mode ID: {mode.mode_id}")
    print(f"   Overtone: n = {mode.n}")
    print(f"   Angular order: l = {mode.l}")
    print(f"   Period: {mode.period:.2f} s")
    print(f"   Reference vp: {mode.vp_ref:.3f} km/s")
    print(f"   Reference vs: {mode.vs_ref:.3f} km/s")
    print(f"   Group velocity: {mode.group_velocity:.2f} km/s")
    print(f"   Volumetric kernel points: {len(mode.vp_depths)}")
    print(f"   Discontinuity kernel points: {len(mode.topo_depths)}")

    # =========================================================================
    # Step 3: Set up function space
    # =========================================================================
    print("\n3. Setting up function space...")
    domain = IntervalDomain(0, 1)
    n_basis = 100
    space = Lebesgue(domain, n_basis=n_basis)

    print(f"   Domain: [0, 1] (normalized depth)")
    print(f"   Basis functions: {n_basis}")
    print(f"   Maps to: [0, {DepthCoordinateSystem.EARTH_RADIUS_KM}] km")

    # =========================================================================
    # Step 4: Create provider
    # =========================================================================
    print("\n4. Creating sensitivity kernel provider...")
    provider = SensitivityKernelProvider(
        catalog,
        space,
        interpolation_method='cubic',
        include_discontinuities=True
    )

    print(f"   Interpolation method: cubic")
    print(f"   Caching enabled: True")

    # =========================================================================
    # Step 5: Get interpolated kernels
    # =========================================================================
    print("\n5. Getting interpolated kernels for mode 00s03...")

    vp_kernel = provider.get_vp_kernel("00s03")
    vs_kernel = provider.get_vs_kernel("00s03")
    rho_kernel = provider.get_rho_kernel("00s03")
    topo_kernel = provider.get_topo_kernel("00s03")

    print(f"   vp kernel shape: {vp_kernel.shape}")
    print(f"   vs kernel shape: {vs_kernel.shape}")
    print(f"   rho kernel shape: {rho_kernel.shape}")
    print(f"   topo kernel: {len(topo_kernel)} discontinuities")

    # =========================================================================
    # Step 6: Visualize kernels
    # =========================================================================
    print("\n6. Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot volumetric kernels
    for ax, param, kernel in zip(
        axes.flat[:3],
        ['vp', 'vs', 'rho'],
        [vp_kernel, vs_kernel, rho_kernel]
    ):
        # Evaluate kernel on fine grid
        x_norm = np.linspace(0, 1, 500)
        kernel_func = space.create_function(kernel)
        y = kernel_func(x_norm)

        # Convert to depth
        x_depth = DepthCoordinateSystem.normalized_to_depth(x_norm)

        # Plot
        ax.plot(x_depth, y, 'b-', linewidth=2)
        ax.set_xlabel('Depth [km]', fontsize=11)
        ax.set_ylabel(f'{param.upper()} Sensitivity', fontsize=11)
        ax.set_title(f'{param.upper()} Kernel', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        # Add major discontinuities
        disc = DepthCoordinateSystem.get_major_discontinuities()
        for name in ['Moho', '410', '660', 'CMB']:
            if name in disc:
                depth, _ = disc[name]
                ax.axvline(x=depth, color='r', linestyle='--',
                          linewidth=0.8, alpha=0.5)
                ax.text(depth, ax.get_ylim()[1]*0.9, name,
                       rotation=90, va='top', ha='right',
                       fontsize=8, color='r', alpha=0.7)

    # Plot discontinuity kernel
    ax = axes[1, 1]
    topo_kernel.plot(ax=ax)
    ax.set_title('Topography Kernel', fontsize=12, fontweight='bold')

    # Add super title
    fig.suptitle(
        f"Sensitivity Kernels - Mode {mode.mode_id} "
        f"(n={mode.n}, l={mode.l}, T={mode.period:.1f}s)",
        fontsize=14, fontweight='bold', y=0.995
    )

    plt.tight_layout()

    # =========================================================================
    # Step 7: Compare multiple modes
    # =========================================================================
    print("\n7. Comparing kernels for multiple modes...")

    # Get some fundamental modes
    fundamental_modes = catalog.list_modes(n_min=0, n_max=0)[:5]
    print(f"   Comparing {len(fundamental_modes)} fundamental modes")

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

    for param, ax in zip(['vp', 'vs', 'rho'], axes2):
        for mode_id in fundamental_modes:
            # Get kernel
            if param == 'vp':
                kernel = provider.get_vp_kernel(mode_id)
            elif param == 'vs':
                kernel = provider.get_vs_kernel(mode_id)
            else:
                kernel = provider.get_rho_kernel(mode_id)

            # Evaluate
            x_norm = np.linspace(0, 1, 200)
            kernel_func = space.create_function(kernel)
            y = kernel_func(x_norm)
            x_depth = DepthCoordinateSystem.normalized_to_depth(x_norm)

            # Plot
            metadata = provider.get_mode_metadata(mode_id)
            label = f"{mode_id} (T={metadata['period']:.0f}s)"
            ax.plot(x_depth, y, label=label, alpha=0.7)

        ax.set_xlabel('Depth [km]')
        ax.set_ylabel(f'{param.upper()} Sensitivity')
        ax.set_title(f'{param.upper()} Kernels Comparison')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.suptitle('Comparison of Fundamental Mode Kernels',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # =========================================================================
    # Step 8: Cache statistics
    # =========================================================================
    print("\n8. Cache statistics...")
    cache_info = provider.get_cache_info()
    print(f"   Modes cached: {cache_info['n_modes_cached']}")
    print(f"   Kernels cached: {cache_info['n_kernels_cached']}")

    # =========================================================================
    # Done
    # =========================================================================
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("Showing plots...")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
